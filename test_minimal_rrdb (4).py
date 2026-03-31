import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import random
import socket
import datetime
import time
import matplotlib.pyplot as plt
import json

# Configuration
config = {
    'hr_dir': "/home/garimella.sri/Super resolution project/SR_dataset/high_resolution",
    'lr_dir': "/home/garimella.sri/Super resolution project/SR_dataset/low_resolution",
    'output_dir': '/home/garimella.sri/Super resolution project/output',
    'batch_size':64,  # Slightly larger batch size 8
    'num_epochs': 5,
    'crop_size': 48,
    'scale': 4,
    'lambda_pixel': 1.0,      # Weight for pixel loss
    'lambda_adv': 0.005,      # Weight for adversarial loss
    'lambda_content': 0.1,    # Weight for content/perceptual loss
}

# Dataset class
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=64, scale=4):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.scale = scale
        
        # Just get a few files to test with
        self.hr_filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:100]
        self.lr_filenames = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:100]
    
    def __len__(self):
        return len(self.hr_filenames)
    
    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_filenames[idx])
        
        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')
        
        # Resize to fixed size to avoid issues
        hr_image = hr_image.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        lr_image = lr_image.resize((self.crop_size // self.scale, self.crop_size // self.scale), Image.BICUBIC)
        
        # Convert to tensor
        hr_tensor = transforms.ToTensor()(hr_image)
        lr_tensor = transforms.ToTensor()(lr_image)
        
        return {'lr': lr_tensor, 'hr': hr_tensor, 'filename': self.hr_filenames[idx]}

# Enhanced Generator with properly connected dense blocks
class Generator(nn.Module):
    def __init__(self, scale=4):
        super(Generator, self).__init__()
        
        # Initial features
        self.conv_first = nn.Conv2d(3, 32, 3, 1, 1)
        
        # Simplified dense blocks (just 2 blocks)
        self.block1 = self._make_dense_block(32)
        self.block2 = self._make_dense_block(32)
        
        # Trunk conv
        self.trunk_conv = nn.Conv2d(32, 32, 3, 1, 1)
        
        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        
        # Final output layer
        self.final = nn.Conv2d(32, 3, 3, 1, 1)
        
    def _make_dense_block(self, nf):
        class DenseBlock(nn.Module):
            def __init__(self, nf):
                super(DenseBlock, self).__init__()
                self.conv1 = nn.Conv2d(nf, 16, 3, 1, 1)
                self.conv2 = nn.Conv2d(nf + 16, 16, 3, 1, 1)
                self.conv3 = nn.Conv2d(nf + 32, 16, 3, 1, 1)
                self.conv4 = nn.Conv2d(nf + 48, nf, 3, 1, 1)
                self.lrelu = nn.LeakyReLU(0.2, inplace=False)
                
            def forward(self, x):
                x1 = self.lrelu(self.conv1(x))
                x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
                x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
                x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
                return torch.add(x4.mul(0.2), x)  # Scaled residual
                
        return DenseBlock(nf)
    
    def forward(self, x):
        feat1 = self.conv_first(x)
        body_out1 = self.block1(feat1)
        body_out2 = self.block2(body_out1)
        trunk = self.trunk_conv(body_out2)
        
        # Use torch.add instead of + operator
        feat2 = torch.add(feat1, trunk)
        
        out = self.upsampling(feat2)
        out = self.final(out)
        
        return out

# Simple discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        return self.net(x)

# VGG Feature Extractor for content loss
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=5, use_bn=False):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pre-trained VGG19
        if use_bn:
            vgg = models.vgg19_bn(weights="IMAGENET1K_V1")
        else:
            vgg = models.vgg19(weights="IMAGENET1K_V1")
            
        # Extract features up to the specified layer (simplified for testing)
        # Layer indices: 0=conv1_1, 2=conv1_2, 5=conv2_1, etc.
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        
        # Freeze parameters to avoid training
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Make sure input is correctly normalized for VGG
        # VGG expects input in the range [0, 1] with normalization
        if x.min().item() < 0:  # If input is in range [-1, 1]
            x = torch.add(torch.mul(x, 0.5), 0.5)  # Convert to [0, 1]
            
        # Apply mean and std normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        return self.features(x)

# Function to save sample images to visualize training progress
def save_sample_images(generator, dataloader, epoch, device, output_dir):
    """Save sample images to visualize training progress"""
    # Create directories
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    generator.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # No need to track gradients
        # Get a batch of images
        batch = next(iter(dataloader))
        lr_imgs = batch['lr'].to(device)
        hr_imgs = batch['hr'].to(device)
        
        # Generate SR images
        sr_imgs = generator(lr_imgs)
        
        # Save a few samples (e.g., first 4 images)
        for i in range(min(4, len(lr_imgs))):
            # Convert tensors to numpy arrays
            lr_img = lr_imgs[i].cpu().numpy().transpose(1, 2, 0)
            hr_img = hr_imgs[i].cpu().numpy().transpose(1, 2, 0)
            sr_img = sr_imgs[i].cpu().numpy().transpose(1, 2, 0)
            
            # Ensure values are in [0, 1] range
            lr_img = np.clip(lr_img, 0, 1)
            hr_img = np.clip(hr_img, 0, 1)
            sr_img = np.clip(sr_img, 0, 1)
            
            # Convert to uint8 for saving
            lr_img = (lr_img * 255).astype(np.uint8)
            hr_img = (hr_img * 255).astype(np.uint8)
            sr_img = (sr_img * 255).astype(np.uint8)
            
            # Create a combined image (LR, SR, HR side by side)
            # Upscale LR image for comparison
            lr_img_upscaled = np.array(Image.fromarray(lr_img).resize(
                (hr_img.shape[1], hr_img.shape[0]), Image.BICUBIC))
            
            # Stack horizontally: LR (upscaled) | SR | HR
            combined = np.concatenate((lr_img_upscaled, sr_img, hr_img), axis=1)
            
            # Save the combined image
            img_path = os.path.join(samples_dir, f'sample_epoch{epoch+1}_img{i+1}.png')
            Image.fromarray(combined).save(img_path)
    
    generator.train()  # Set model back to training mode
    
    return os.path.join(samples_dir, f'sample_epoch{epoch+1}_img1.png')  # Return path to first sample

# Function to create performance plots
def create_performance_plots(results_file, output_dir):
    """Create performance plots from results file"""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    gpu_counts = sorted(list(results.keys()))
    epoch_times = [results[str(gpu)]['avg_epoch_time'] for gpu in gpu_counts]
    speedups = [results['1']['avg_epoch_time'] / results[str(gpu)]['avg_epoch_time'] for gpu in gpu_counts]
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    # Epoch time plot
    plt.subplot(1, 2, 1)
    plt.bar(gpu_counts, epoch_times)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Average Epoch Time (s)')
    plt.title('Training Time vs. GPU Count')
    
    # Speedup plot
    plt.subplot(1, 2, 2)
    plt.bar(gpu_counts, speedups)
    plt.plot(gpu_counts, gpu_counts, 'r--', label='Ideal Linear Speedup')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Speedup (X)')
    plt.title('Scaling Efficiency')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'performance_scaling.png')
    plt.savefig(plot_path)
    
    return plot_path

# Initialize distributed with file store for more reliable initialization
def init_distributed(rank, world_size):
    # Use a common store file
    store_dir = os.path.join(config['output_dir'], "torch_ddp_store")
    os.makedirs(store_dir, exist_ok=True)
    
    store_file = os.path.join(store_dir, f"store_{int(time.time())}")
    if rank == 0:
        # Master creates the file
        open(store_file, 'w').close()
        print(f"Rank {rank} created store file: {store_file}")
    
    # Wait to make sure file is created
    time.sleep(2)
    
    # Initialize with file store (more reliable than TCP)
    print(f"Rank {rank}: Initializing with file store: {store_file}")
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{store_file}",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=120)
        )
        
        # Set device
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Successfully initialized process group")
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group: {e}")
        raise

# Training process
def train_process(rank, world_size, config):
    try:
        print(f"Rank {rank}: Starting process")
        
        # Initialize distributed
        init_distributed(rank, world_size)
        print(f"Rank {rank}: Process initialized")
        
        # Models
        generator = Generator().to(rank)
        discriminator = Discriminator().to(rank)
        vgg_extractor = VGGFeatureExtractor().to(rank)  # VGG for content loss
        
        # Wrap with DDP (only generator and discriminator - not VGG)
        generator = DDP(generator, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])
        
        # Loss functions
        pixel_criterion = nn.MSELoss()  # For pixel-wise loss
        adv_criterion = nn.MSELoss()    # For adversarial loss
        content_criterion = nn.L1Loss()  # For content loss
        
        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
        
        # Dataset
        dataset = SRDataset(
            config['hr_dir'],
            config['lr_dir'],
            crop_size=config['crop_size'],
            scale=config['scale']
        )
        
        # Sampler and dataloader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
        
        # Training
        generator.train()
        discriminator.train()
        
        total_time = 0.0
        epoch_times = []
        
        for epoch in range(config['num_epochs']):
            # Start timing for this epoch
            epoch_start_time = time.time()
            
            sampler.set_epoch(epoch)
            
            batch_times = []
            g_forward_times = []
            g_backward_times = []
            d_forward_times = []
            d_backward_times = []
            
            for batch_idx, batch in enumerate(dataloader):
                batch_start_time = time.time()
                
                lr_imgs = batch['lr'].to(rank)
                hr_imgs = batch['hr'].to(rank)
                
                #-----------------------
                # Train Generator
                #-----------------------
                optimizer_G.zero_grad()
                
                # Time generator forward pass
                g_forward_start = time.time()
                gen_imgs = generator(lr_imgs)
                
                # 1. Pixel loss
                pixel_loss = pixel_criterion(gen_imgs, hr_imgs)
                
                # 2. Content loss (VGG feature matching)
                hr_features = vgg_extractor(hr_imgs)
                sr_features = vgg_extractor(gen_imgs)
                content_loss = content_criterion(sr_features, hr_features)
                
                # 3. Adversarial loss (relativistic)
                real_validity = discriminator(hr_imgs)
                fake_validity = discriminator(gen_imgs)
                adv_loss = adv_criterion(fake_validity - torch.mean(real_validity), torch.ones_like(fake_validity))
                g_forward_end = time.time()
                g_forward_times.append(g_forward_end - g_forward_start)
                
                # Combined generator loss
                g_loss = (config['lambda_pixel'] * pixel_loss + 
                         config['lambda_content'] * content_loss + 
                         config['lambda_adv'] * adv_loss)
                
                # Time generator backward pass
                g_backward_start = time.time()
                g_loss.backward()
                optimizer_G.step()
                g_backward_end = time.time()
                g_backward_times.append(g_backward_end - g_backward_start)
                
                #-----------------------
                # Train Discriminator
                #-----------------------
                optimizer_D.zero_grad()
                
                # Detach generated images to avoid training generator through discriminator
                with torch.no_grad():
                    fake_imgs = generator(lr_imgs).detach()
                
                # Time discriminator forward pass
                d_forward_start = time.time()
                # Discriminator outputs
                real_validity = discriminator(hr_imgs)
                fake_validity = discriminator(fake_imgs)
                
                # Relativistic discriminator loss
                real_loss = adv_criterion(real_validity - torch.mean(fake_validity), torch.ones_like(real_validity))
                fake_loss = adv_criterion(fake_validity - torch.mean(real_validity), torch.zeros_like(fake_validity))
                d_loss = (real_loss + fake_loss) * 0.5
                d_forward_end = time.time()
                d_forward_times.append(d_forward_end - d_forward_start)
                
                # Time discriminator backward pass
                d_backward_start = time.time()
                d_loss.backward()
                optimizer_D.step()
                d_backward_end = time.time()
                d_backward_times.append(d_backward_end - d_backward_start)
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                
                # Print progress
                if rank == 0 and batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, "
                         f"G Loss: {g_loss.item():.4f} (Pixel: {pixel_loss.item():.4f}, "
                         f"Content: {content_loss.item():.4f}, Adv: {adv_loss.item():.4f}), "
                         f"D Loss: {d_loss.item():.4f}, "
                         f"Batch Time: {batch_time:.4f}s")
                
                # Clean memory
                torch.cuda.empty_cache()
                
                # Just train on a few batches for testing
                if batch_idx > 20:
                    break
            
            # End timing for this epoch
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            total_time += epoch_duration
            
            # Generate and save sample images
            if rank == 0:
                sample_path = save_sample_images(generator, dataloader, epoch, rank, config['output_dir'])
                print(f"Saved sample images for epoch {epoch+1} at: {sample_path}")
            
            # Calculate timing statistics
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            avg_g_forward = sum(g_forward_times) / len(g_forward_times) if g_forward_times else 0
            avg_g_backward = sum(g_backward_times) / len(g_backward_times) if g_backward_times else 0
            avg_d_forward = sum(d_forward_times) / len(d_forward_times) if d_forward_times else 0
            avg_d_backward = sum(d_backward_times) / len(d_backward_times) if d_backward_times else 0
            
            # Print timing information
            if rank == 0:
                print(f"\n--- Timing Statistics for Epoch {epoch+1} ---")
                print(f"Epoch Duration: {epoch_duration:.2f} seconds")
                print(f"Average Batch Time: {avg_batch_time:.4f} seconds")
                print(f"Average Generator Forward Time: {avg_g_forward:.4f} seconds")
                print(f"Average Generator Backward Time: {avg_g_backward:.4f} seconds")
                print(f"Average Discriminator Forward Time: {avg_d_forward:.4f} seconds")
                print(f"Average Discriminator Backward Time: {avg_d_backward:.4f} seconds")
                print(f"Time Breakdown: G-Forward ({avg_g_forward/avg_batch_time*100:.1f}%), "
                     f"G-Backward ({avg_g_backward/avg_batch_time*100:.1f}%), "
                     f"D-Forward ({avg_d_forward/avg_batch_time*100:.1f}%), "
                     f"D-Backward ({avg_d_backward/avg_batch_time*100:.1f}%), "
                     f"Other ({(1-avg_g_forward/avg_batch_time-avg_g_backward/avg_batch_time-avg_d_forward/avg_batch_time-avg_d_backward/avg_batch_time)*100:.1f}%)")
                
                # Safe division to avoid zero division error
                completed_epochs = epoch + 1  # epochs are 0-indexed
                avg_epoch_time = total_time / completed_epochs
                remaining_epochs = config['num_epochs'] - completed_epochs
                estimated_total = total_time + (avg_epoch_time * remaining_epochs)
                
                print(f"Completed {completed_epochs}/{config['num_epochs']} epochs")
                print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
                print(f"Estimated total training time: {estimated_total:.2f} seconds")
                print("-------------------------------------------\n")
        
        # Save performance metrics for this run
        if rank == 0:
            # Calculate final performance metrics
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            
            # Append to results file
            results_file = os.path.join(config['output_dir'], 'performance_results.json')
            
            # Load existing results if any
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError:
                        results = {}
            else:
                results = {}
            
            # Add results for this GPU count
            results[str(world_size)] = {
                'avg_epoch_time': avg_epoch_time,
                'total_time': total_time,
                'epochs': config['num_epochs'],
                'batch_size': config['batch_size'],
                'epoch_times': [float(t) for t in epoch_times]
            }
            
            # Save updated results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create performance plots if we have at least two data points
            if len(results.keys()) >= 2:
                plot_path = create_performance_plots(results_file, config['output_dir'])
                print(f"Created performance plots at: {plot_path}")
    
        # Clean up process group
        try:
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed successfully")
        except Exception as e:
            print(f"Rank {rank}: Error destroying process group: {e}")
            
    except Exception as e:
        # Log any errors
        import traceback
        print(f"Rank {rank}: Error during training: {e}")
        print(traceback.format_exc())
        
        # Try to clean up
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass

def main(world_size):
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    if world_size == 1:
        train_process(0, 1, config)
    else:
        mp.spawn(
            train_process,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    args = parser.parse_args()
    
    # Override batch size if provided
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        
    main(args.world_size)