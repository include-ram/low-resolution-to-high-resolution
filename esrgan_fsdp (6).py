import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
import random
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration optimized for 4-GPU performance with better scaling
config = {
    'hr_dir': "/home/garimella.sri/Super resolution project/SR_dataset/high_resolution",
    'lr_dir': "/home/garimella.sri/Super resolution project/SR_dataset/low_resolution",
    'output_dir': '/home/garimella.sri/Super resolution project/output',
    'scale': 4,
    'batch_size': 4,  # Smaller batch size for reduced memory footprint
    'num_epochs': 5,
    'lr_G': 1e-4,
    'lr_D': 1e-4,
    'b1': 0.9,
    'b2': 0.999,
    'num_workers': 0,  # Set to 0 for more predictable timing
    'vgg_feature_layer': 5,  # Use very shallow VGG for better performance
    'lambda_adv': 5e-3,
    'lambda_pixel': 1e-2,
    'lambda_content': 1.0,
    'crop_size': 48,
    
    # Core optimization parameters
    'gradient_accumulation_steps': {
        1: 2,   # 1 GPU: 2 accumulation steps
        2: 4,   # 2 GPUs: 4 accumulation steps
        4: 8,   # 4 GPUs: 8 accumulation steps
    },
    'activation_checkpointing': True,
    
    # Model parameters
    'generator_blocks': 4,  # Number of RRDB blocks in the generator
    'generator_channels': 64,  # Number of channels in the generator
    'discriminator_channels': 32,  # Number of initial channels in the discriminator
    
    # Benchmarking parameters
    'benchmark_iters': 10,
    'warmup_iters': 3,
}

# Set NCCL environment variables for better performance
def set_nccl_env():
    """Set NCCL environment variables for better performance"""
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_NTHREADS'] = '4'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    os.environ['NCCL_BUFFSIZE'] = '2097152'  # 2MB buffer size
    
    # Tune channels for better throughput
    os.environ['NCCL_MIN_NCHANNELS'] = '4'
    
    # Set algorithm based on GPU count
    if torch.cuda.device_count() >= 4:
        os.environ['NCCL_ALGO'] = 'RING'
    else:
        os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Force ring for small GPU counts

# Initialize process group with optimized parameters
def init_distributed(world_size, rank, port=12355, timeout=600):
    """Initialize process group with optimized settings for multiple GPUs"""
    print(f"Initializing process group: rank {rank} / world_size {world_size}")
    
    # Set NCCL environment variables
    set_nccl_env()
    
    # Set PyTorch environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=timeout)
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 on Ampere GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set seed for reproducibility
    torch.manual_seed(42 + rank)
    
    print(f"Rank {rank}: Initialized distributed process group")

# Custom Dataset with efficient loading
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=64, scale=4, transform=None, cache_size=50):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.crop_size = crop_size
        self.scale = scale
        self.cache_size = cache_size
        self.cache = {}
        
        # Get image filenames
        self.hr_filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_filenames = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure matching files
        assert len(self.hr_filenames) == len(self.lr_filenames), "Number of HR and LR images don't match"
    
    def __len__(self):
        return len(self.hr_filenames)
    
    def __getitem__(self, idx):
        # Check if in cache first
        if idx in self.cache:
            return self.cache[idx]
            
        hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_filenames[idx])
        
        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')
        
        # Direct resize for simplicity and consistency
        hr_image = hr_image.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        lr_image = lr_image.resize((self.crop_size // self.scale, self.crop_size // self.scale), Image.BICUBIC)
        
        # Convert to tensor
        if self.transform:
            hr_tensor = self.transform(hr_image)
            lr_tensor = self.transform(lr_image)
        else:
            hr_tensor = transforms.ToTensor()(hr_image)
            lr_tensor = transforms.ToTensor()(lr_image)
        
        result = {'lr': lr_tensor, 'hr': hr_tensor, 'filename': self.hr_filenames[idx]}
        
        # Add to cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[idx] = result
            
        return result

# Simplified and efficient RDB block
class EfficientRDB(nn.Module):
    def __init__(self, channels=64, growth_channels=32, scale=0.2):
        super(EfficientRDB, self).__init__()
        self.scale = scale
        
        # Simplified RDB with fewer convolutions
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels + growth_channels, channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.conv2(torch.cat((x, x1), 1))
        return x2 * self.scale + x

# RRDB module with fewer blocks
class RRDB(nn.Module):
    def __init__(self, channels=64, growth_channels=32, scale=0.2):
        super(RRDB, self).__init__()
        self.scale = scale
        self.rdb1 = EfficientRDB(channels, growth_channels)
        self.rdb2 = EfficientRDB(channels, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        return out * self.scale + x

# Efficient generator with fewer operations
class EfficientGenerator(nn.Module):
    def __init__(self, num_blocks=4, channels=64, growth_channels=32, scale=4):
        super(EfficientGenerator, self).__init__()
        self.scale = scale
        self.channels = channels
        
        # First convolution
        self.conv_first = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        
        # RRDB blocks
        rrdb_blocks = []
        for _ in range(num_blocks):
            rrdb_blocks.append(RRDB(channels, growth_channels))
        self.RRDB_trunk = nn.Sequential(*rrdb_blocks)
        
        # Trunk convolution
        self.trunk_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        # Direct upsampling (no intermediate blocks)
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        # First convolution
        fea = self.conv_first(x)
        
        # RRDB trunk
        trunk = self.RRDB_trunk(fea)
        trunk = self.trunk_conv(trunk)
        
        # Global residual learning
        fea = fea + trunk
        
        # Upsampling
        out = self.upsampling(fea)
        
        return out

# Efficient discriminator with fewer layers
class EfficientDiscriminator(nn.Module):
    def __init__(self, input_size=48, base_channels=32):
        super(EfficientDiscriminator, self).__init__()
        
        # Simplified discriminator architecture
        self.model = nn.Sequential(
            # Initial layer
            nn.Conv2d(3, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Middle layers with group norm (more efficient than batch norm)
            nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1),
            nn.GroupNorm(4, base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layers
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*4, 1)
        )
        
    def forward(self, x):
        return self.model(x)

# Simplified perceptual loss with minimal VGG
class EfficientPerceptualLoss(nn.Module):
    def __init__(self, feature_layer=5):
        super(EfficientPerceptualLoss, self).__init__()
        from torchvision import models
        
        # Use minimal VGG
        vgg = models.vgg11(weights="IMAGENET1K_V1")
        
        # Only use first few layers
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()
        
    def forward(self, sr, hr):
        # Cast to float32 for VGG
        sr = sr.to(torch.float32)
        hr = hr.to(torch.float32)
        
        # Extract features
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        
        # Calculate loss
        return self.criterion(sr_features, hr_features)

# Apply activation checkpointing to model
def apply_activation_checkpointing(model):
    """Apply checkpointing to specific modules for memory savings"""
    def create_checkpoint_wrapper(module):
        # Apply checkpointing to RRDB modules
        if isinstance(module, RRDB):
            return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        return module
    
    # Apply recursively
    for name, child in model.named_children():
        if isinstance(child, nn.Sequential) and any(isinstance(m, RRDB) for m in child):
            # If it's a sequential containing RRDBs, wrap the sequence
            setattr(model, name, nn.Sequential(*[create_checkpoint_wrapper(m) for m in child]))
        elif isinstance(child, RRDB):
            # Directly wrap RRDB modules
            setattr(model, name, create_checkpoint_wrapper(child))
        elif list(child.children()):
            # Recursively apply to all children modules
            apply_activation_checkpointing(child)
    
    return model

# Log GPU memory usage
def log_gpu_memory(rank, label=""):
    """Log GPU memory usage at a specific stage"""
    if rank == 0:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(rank) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(rank) / (1024 ** 3)
        print(f"GPU Memory [{label}] - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Benchmark process for multi-GPU with FSDP
def benchmark_process(rank, world_size, benchmark_config, results_dict=None):
    """Benchmark process for each GPU with optimized FSDP configuration"""
    # Initialize distributed
    port = int(os.environ.get('MASTER_PORT', 12355))
    init_distributed(world_size, rank, port)
    
    print(f"Rank {rank}: Starting benchmark process")
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    
    print(f"Rank {rank}: Using device: {device}")
    
    # Get gradient accumulation steps for this world size
    grad_accum_steps = benchmark_config['gradient_accumulation_steps'].get(world_size, 1)
    
    # Create generator and discriminator
    generator = EfficientGenerator(
        num_blocks=benchmark_config['generator_blocks'],
        channels=benchmark_config['generator_channels'],
        scale=benchmark_config['scale']
    ).to(device)
    
    discriminator = EfficientDiscriminator(
        input_size=benchmark_config['crop_size'],
        base_channels=benchmark_config['discriminator_channels']
    ).to(device)
    
    # Apply activation checkpointing if enabled
    if benchmark_config['activation_checkpointing']:
        generator = apply_activation_checkpointing(generator)
    
    # Select appropriate sharding strategy
    sharding_strategy = ShardingStrategy.NO_SHARD
    if world_size > 1:
        if world_size == 2:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # FSDP wrapper args - minimal for compatibility
    fsdp_kwargs = {
        "device_id": rank,
        "sharding_strategy": sharding_strategy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    }
    
    # Wrap models with FSDP
    generator = FSDP(generator, **fsdp_kwargs)
    discriminator = FSDP(discriminator, **fsdp_kwargs)
    
    log_gpu_memory(rank, "Models Created")
    
    # Loss functions
    content_criterion = EfficientPerceptualLoss(feature_layer=benchmark_config['vgg_feature_layer']).to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    pixel_criterion = nn.L1Loss().to(device)
    
    # Optimizers with float32 parameters for stability
    optimizer_G = optim.AdamW(
        [p.float() if p.dtype == torch.float16 else p for p in generator.parameters()], 
        lr=benchmark_config['lr_G'], 
        betas=(benchmark_config['b1'], benchmark_config['b2']), 
        eps=1e-4,
        weight_decay=1e-5
    )
    
    optimizer_D = optim.AdamW(
        [p.float() if p.dtype == torch.float16 else p for p in discriminator.parameters()], 
        lr=benchmark_config['lr_D'], 
        betas=(benchmark_config['b1'], benchmark_config['b2']), 
        eps=1e-4,
        weight_decay=1e-5
    )
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = SRDataset(
        benchmark_config['hr_dir'], 
        benchmark_config['lr_dir'], 
        crop_size=benchmark_config['crop_size'],
        scale=benchmark_config['scale'],
        transform=transform,
        cache_size=50
    )
    
    # Use DistributedSampler for data sharding
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # Per GPU batch size
    per_gpu_batch_size = benchmark_config['batch_size']
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=0,  # Use 0 for better timing consistency
        pin_memory=True,
        drop_last=True,
    )
    
    log_gpu_memory(rank, "Before Warmup")
    
    # Warmup phase
    if rank == 0:
        print("Warming up...")
    
    generator.train()
    discriminator.train()
    
    # Pre-load batches for deterministic timing
    preloaded_batches = []
    warmup_iters = min(benchmark_config['warmup_iters'], len(dataloader))
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < warmup_iters + benchmark_config['benchmark_iters']:
            preloaded_batches.append(batch)
        else:
            break
    
    # Simplified warmup loop
    for batch_idx in range(warmup_iters):
        batch = preloaded_batches[batch_idx]
        lr_imgs = batch['lr'].to(device, non_blocking=True)
        hr_imgs = batch['hr'].to(device, non_blocking=True)
        
        # Skip mismatched sizes
        if lr_imgs.size(2) * benchmark_config['scale'] != hr_imgs.size(2) or lr_imgs.size(3) * benchmark_config['scale'] != hr_imgs.size(3):
            continue
        
        # Simple forward-backward for warmup
        # Generator forward-backward
        optimizer_G.zero_grad(set_to_none=True)
        gen_imgs = generator(lr_imgs)
        g_loss = pixel_criterion(gen_imgs, hr_imgs)
        g_loss.backward()
        optimizer_G.step()
        
        # Discriminator forward-backward
        optimizer_D.zero_grad(set_to_none=True)
        d_loss = adversarial_criterion(
            discriminator(hr_imgs),
            torch.ones(hr_imgs.size(0), 1, device=device)
        )
        d_loss.backward()
        optimizer_D.step()
    
    log_gpu_memory(rank, "After Warmup")
    
    # Clear cache before benchmark
    torch.cuda.empty_cache()
    
    # Synchronize before benchmark
    torch.cuda.synchronize()
    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Barrier failed: {e}")
    
    # Start benchmark
    if rank == 0:
        print("Starting benchmark...")
    
    # Benchmark with CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Reset accumulation counter
    accumulated_steps = 0
    
    # Record start time
    start_event.record()
    
    # Benchmark loop
    for batch_idx in range(warmup_iters, warmup_iters + benchmark_config['benchmark_iters']):
        # Get batch
        batch = preloaded_batches[batch_idx]
        lr_imgs = batch['lr'].to(device, non_blocking=True)
        hr_imgs = batch['hr'].to(device, non_blocking=True)
        
        # Skip mismatched sizes
        if lr_imgs.size(2) * benchmark_config['scale'] != hr_imgs.size(2) or lr_imgs.size(3) * benchmark_config['scale'] != hr_imgs.size(3):
            continue
        
        # Only sync gradients at accumulation boundaries
        sync_grads = (accumulated_steps + 1) % grad_accum_steps == 0
        
        # Generator forward/backward with accumulation
        # Only zero grads at start of accumulation
        if accumulated_steps % grad_accum_steps == 0:
            optimizer_G.zero_grad(set_to_none=True)
        
        # Forward pass
        gen_imgs = generator(lr_imgs)
        
        # Calculate losses
        pixel_loss = pixel_criterion(gen_imgs, hr_imgs)
        content_loss = content_criterion(gen_imgs, hr_imgs)
        
        # Discriminator predictions
        real_preds = discriminator(hr_imgs)
        fake_preds = discriminator(gen_imgs)
        
        # Adversarial loss (relativistic)
        adv_loss = adversarial_criterion(
            fake_preds - torch.mean(real_preds),
            torch.ones_like(fake_preds)
        )
        
        # Combined generator loss (scaled for accumulation)
        g_loss = (
            benchmark_config['lambda_pixel'] * pixel_loss +
            benchmark_config['lambda_content'] * content_loss +
            benchmark_config['lambda_adv'] * adv_loss
        ) / grad_accum_steps
        
        # Backward
        g_loss.backward()
        
        # Step optimizer at accumulation boundaries
        if sync_grads:
            optimizer_G.step()
        
        # Discriminator forward/backward with accumulation
        # Only zero grads at start of accumulation
        if accumulated_steps % grad_accum_steps == 0:
            optimizer_D.zero_grad(set_to_none=True)
        
        # Forward pass with detached generator output
        real_preds = discriminator(hr_imgs)
        fake_preds = discriminator(gen_imgs.detach())
        
        # Relativistic discriminator loss
        real_loss = adversarial_criterion(
            real_preds - torch.mean(fake_preds),
            torch.ones_like(real_preds)
        )
        
        fake_loss = adversarial_criterion(
            fake_preds - torch.mean(real_preds),
            torch.zeros_like(fake_preds)
        )
        
        # Combined discriminator loss (scaled for accumulation)
        d_loss = (real_loss + fake_loss) / 2 / grad_accum_steps
        
        # Backward
        d_loss.backward()
        
        # Step optimizer at accumulation boundaries
        if sync_grads:
            optimizer_D.step()
        
        # Update accumulation counter
        accumulated_steps += 1
    
    # Apply any remaining gradient updates
    if accumulated_steps % grad_accum_steps != 0:
        optimizer_G.step()
        optimizer_D.step()
    
    # Record end time and synchronize
    end_event.record()
    torch.cuda.synchronize()
    
    # Final synchronization
    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Final barrier failed: {e}")
    
    log_gpu_memory(rank, "After Benchmark")
    
    # Calculate elapsed time (ms to seconds)
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_seconds = elapsed_ms / 1000.0
    
    # Calculate performance metrics
    if rank == 0:
        # Calculate effective batch sizes
        effective_per_gpu_batch = per_gpu_batch_size * grad_accum_steps
        effective_global_batch = effective_per_gpu_batch * world_size
        
        # Calculate throughput
        total_batch_size = per_gpu_batch_size * world_size
        total_images = benchmark_config['benchmark_iters'] * total_batch_size
        images_per_sec = total_images / elapsed_seconds
        per_gpu_throughput = images_per_sec / world_size
        
        # Store result for this GPU count
        if results_dict is not None:
            results_dict[world_size] = {
                'time_seconds': elapsed_seconds,
                'num_batches': benchmark_config['benchmark_iters'],
                'total_batch_size': total_batch_size,
                'per_gpu_batch_size': per_gpu_batch_size,
                'gradient_accumulation_steps': grad_accum_steps,
                'effective_per_gpu_batch': effective_per_gpu_batch,
                'effective_global_batch': effective_global_batch,
                'images_per_second': images_per_sec,
                'images_per_second_per_gpu': per_gpu_throughput
            }
        
        print(f"Time for {benchmark_config['benchmark_iters']} batches: {elapsed_seconds:.2f} seconds")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective global batch size: {effective_global_batch}")
        print(f"Throughput: {images_per_sec:.2f} images/second with {world_size} GPUs")
        print(f"Per-GPU throughput: {per_gpu_throughput:.2f} images/second/GPU")
    
    # Clean up
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except:
            if rank == 0:
                print("Warning: Error during cleanup")

# Create performance visualization plots
def create_performance_plots(results, output_dir):
    """Create visualizations from benchmark results"""
    if len(results) <= 1:
        print("Not enough data points for visualization.")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data
    gpu_counts = sorted(results.keys())
    throughputs = [results[g]['images_per_second'] for g in gpu_counts]
    speedups = [results[g]['images_per_second'] / results[1]['images_per_second'] for g in gpu_counts]
    times = [results[g]['time_seconds'] for g in gpu_counts]
    batch_times = [results[g]['time_seconds'] / results[g]['num_batches'] for g in gpu_counts]
    effective_batches = [results[g]['effective_global_batch'] for g in gpu_counts]
    
    # Create detailed visualization
    plt.figure(figsize=(15, 10))
    
    # 1. Throughput
    plt.subplot(2, 3, 1)
    plt.bar(range(len(gpu_counts)), throughputs, color='skyblue')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Images per Second')
    plt.title('Total Throughput')
    for i, v in enumerate(throughputs):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    
    # 2. Speedup with ideal scaling line
    plt.subplot(2, 3, 2)
    plt.bar(range(len(gpu_counts)), speedups, color='lightgreen')
    plt.plot(range(len(gpu_counts)), gpu_counts, 'r--', label='Ideal Linear')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Speedup vs. 1 GPU')
    plt.title('Scaling Efficiency')
    plt.legend()
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    # 3. Scaling efficiency percentage
    plt.subplot(2, 3, 3)
    efficiency = [s / g * 100 for s, g in zip(speedups, gpu_counts)]
    plt.bar(range(len(gpu_counts)), efficiency, color='salmon')
    plt.axhline(y=100, color='r', linestyle='--', label='Ideal (100%)')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Efficiency (%)')
    plt.title('Scaling Efficiency Percentage')
    plt.legend()
    for i, v in enumerate(efficiency):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # 4. Per-GPU throughput
    plt.subplot(2, 3, 4)
    per_gpu_throughput = [t / g for t, g in zip(throughputs, gpu_counts)]
    plt.bar(range(len(gpu_counts)), per_gpu_throughput, color='mediumpurple')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Images/second/GPU')
    plt.title('Per-GPU Throughput')
    for i, v in enumerate(per_gpu_throughput):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # 5. Batch processing time
    plt.subplot(2, 3, 5)
    plt.bar(range(len(gpu_counts)), batch_times, color='gold')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Seconds per Batch')
    plt.title('Batch Processing Time')
    for i, v in enumerate(batch_times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # 6. Effective batch size
    plt.subplot(2, 3, 6)
    plt.bar(range(len(gpu_counts)), effective_batches, color='lightcoral')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Images')
    plt.title('Effective Global Batch Size')
    for i, v in enumerate(effective_batches):
        plt.text(i, v + 1, f"{v}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scaling_analysis.png'), dpi=300)
    plt.close()
    
    # Create simplified plot for throughput and scaling
    plt.figure(figsize=(12, 5))
    
    # Throughput plot
    plt.subplot(1, 2, 1)
    plt.bar(range(len(gpu_counts)), throughputs, color='skyblue')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Images per Second')
    plt.title('Throughput vs. GPU Count')
    for i, v in enumerate(throughputs):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    
    # Speedup plot
    plt.subplot(1, 2, 2)
    plt.bar(range(len(gpu_counts)), speedups, color='lightgreen')
    plt.plot(range(len(gpu_counts)), gpu_counts, 'r--', label='Ideal Scaling')
    plt.xticks(range(len(gpu_counts)), [f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    plt.ylabel('Speedup (relative to 1 GPU)')
    plt.ylim(0, max(gpu_counts[-1], max(speedups)) * 1.1)
    plt.title('Scaling Efficiency')
    plt.legend()
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scaling_results.png'), dpi=300)
    plt.close()
    
    return os.path.join(plots_dir, 'scaling_analysis.png')

# Benchmark multi-GPU function
def benchmark_multi_gpu():
    """Benchmark performance with different GPU configurations"""
    results = {}
    
    # Ensure we have enough GPUs
    max_gpus = min(4, torch.cuda.device_count())
    
    # Create benchmark directory
    benchmark_dir = os.path.join(config['output_dir'], 'benchmark_optimized')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Run benchmarks for different GPU configurations
    for num_gpus in [1, 2, 4]:
        if num_gpus > max_gpus:
            print(f"Skipping {num_gpus} GPU benchmark (only {max_gpus} GPUs available)")
            continue
            
        print(f"\nBenchmarking with {num_gpus} GPUs...")
        
        # Create a copy of config for this run
        benchmark_config = config.copy()
        
        # Update output directory for this run
        benchmark_config['output_dir'] = os.path.join(benchmark_dir, f'{num_gpus}_gpu')
        os.makedirs(benchmark_config['output_dir'], exist_ok=True)
        
        # Use a different port for each configuration
        port = 30000 + random.randint(0, 10000)
        os.environ['MASTER_PORT'] = str(port)
        print(f"Using port {port} for {num_gpus} GPU benchmark")
        
        # Multi-GPU uses spawn
        if num_gpus > 1:
            mp.spawn(
                benchmark_process,
                args=(num_gpus, benchmark_config, results),
                nprocs=num_gpus,
                join=True
            )
        else:
            # Single GPU direct call
            benchmark_process(0, 1, benchmark_config, results)
        
        # Clean up between runs
        torch.cuda.empty_cache()
    
    # Print comparison
    print("\nBenchmark Results:")
    print("=" * 100)
    print(f"{'GPUs':<5} {'Batch Size':<10} {'Grad Accum':<10} {'Eff. Batch':<10} {'Images/sec':<15} {'Time (s)':<10} {'Speedup':<10}")
    print("-" * 100)
    
    baseline = results.get(1, {}).get('images_per_second', 1.0)
    
    for num_gpus in sorted(results.keys()):
        result = results[num_gpus]
        speedup = result['images_per_second'] / baseline
        
        print(f"{num_gpus:<5} {result['total_batch_size']:<10} "
              f"{result['gradient_accumulation_steps']:<10} "
              f"{result['effective_global_batch']:<10} "
              f"{result['images_per_second']:<15.2f} {result['time_seconds']:<10.2f} "
              f"{speedup:<10.2f}x")
    
    print("=" * 100)
    
    # Save results to file
    with open(os.path.join(benchmark_dir, 'benchmark_results.txt'), 'w') as f:
        f.write("Benchmark Results:\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'GPUs':<5} {'Batch Size':<10} {'Grad Accum':<10} {'Eff. Batch':<10} {'Images/sec':<15} {'Time (s)':<10} {'Speedup':<10}\n")
        f.write("-" * 100 + "\n")
        
        for num_gpus in sorted(results.keys()):
            result = results[num_gpus]
            speedup = result['images_per_second'] / baseline
            
            f.write(f"{num_gpus:<5} {result['total_batch_size']:<10} "
                  f"{result['gradient_accumulation_steps']:<10} "
                  f"{result['effective_global_batch']:<10} "
                  f"{result['images_per_second']:<15.2f} {result['time_seconds']:<10.2f} "
                  f"{speedup:<10.2f}x\n")
        
        f.write("=" * 100 + "\n")
    
    # Create performance visualization plots
    if len(results) > 1:
        plot_path = create_performance_plots(results, benchmark_dir)
        print(f"Created performance visualization at {plot_path}")
    
    return results

# Main function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized ESRGAN with FSDP training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark instead of training')
    parser.add_argument('--batch_size', type=int, help='Per-GPU batch size (overrides config)')
    args = parser.parse_args()
    
    # Override batch size if provided
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    if args.benchmark:
        benchmark_multi_gpu()
    else:
        print("Please use --benchmark flag to run the benchmark")