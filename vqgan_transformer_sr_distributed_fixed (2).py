import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import sys
import re
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
import argparse

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add argument parser
parser = argparse.ArgumentParser(description='VQGAN-Transformer Super Resolution Distributed Training')
parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--master_port', type=str, default='29501', help='Master port for distributed training')
parser.add_argument('--local_rank', '--local-rank', type=int, default=0, help='Local rank for distributed training')
parser.add_argument('--hr_dir', type=str, default="/home/garimella.sri/Super resolution project/SR_dataset/high_resolution", 
                    help='Directory containing high-resolution images')
parser.add_argument('--lr_dir', type=str, default="/home/garimella.sri/Super resolution project/SR_dataset/low_resolution", 
                    help='Directory containing low-resolution images')
parser.add_argument('--output_dir', type=str, default="/home/garimella.sri/Super resolution project/output/transformer", 
                    help='Output directory')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
parser.add_argument('--vqgan_epochs', type=int, default=100, help='Number of epochs to train VQGAN')
parser.add_argument('--transformer_epochs', type=int, default=80, help='Number of epochs to train Transformer')
parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
args = parser.parse_args()

# Handle the case where local_rank is passed as --local-rank=N
for i, arg in enumerate(sys.argv):
    if arg.startswith('--local-rank='):
        local_rank_value = int(arg.split('=')[1])
        args.local_rank = local_rank_value

# Set parameters
config = {
    # Data parameters
    'hr_dir': args.hr_dir,
    'lr_dir': args.lr_dir,
    'output_dir': args.output_dir,
    'checkpoint_dir': os.path.join(args.output_dir, "checkpoints"),
    
    # Model parameters
    'low_res': 64,
    'high_res': 256,
    'hidden_dims': 32,  # Reduced from 64 for memory constraints
    'codebook_size': 1024,
    'embedding_dim': 128,  # Reduced from 256 for memory constraints
    'transformer_dim': 128,  # Reduced from 256 for memory constraints
    'transformer_depth': 4,
    'transformer_heads': 8,
    'transformer_dropout': 0.1,
    
    # Training parameters
    'batch_size': 1,  # Per GPU
    'vqgan_epochs': 1,
    'transformer_epochs':1,
    'vqgan_lr': 2e-4,
    'transformer_lr': 1e-4,
    'num_workers': 4*args.num_workers,
    
    # Checkpoint saving frequency
    'save_checkpoint_every': 1,  # Save every 10 epochs
    'evaluate_every': 1,         # Show evaluation images every 5 epochs
}

# Modified init_distributed function with better environment detection
def init_distributed():
    # Check if using torch.distributed.launch or torchrun
    if 'LOCAL_RANK' in os.environ or 'RANK' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        
        print(f"Distributed environment detected: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Initialize the process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, local_rank, world_size, rank
    # Manual setting via arguments
    elif args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.local_rank)
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        
        local_rank = args.local_rank
        world_size = args.world_size
        rank = args.local_rank
        
        print(f"Manual distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Initialize the process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, local_rank, world_size, rank
    else:
        print("Distributed training not enabled. Will run on single GPU.")
        return False, 0, 1, 0

# Clear CUDA cache before starting
torch.cuda.empty_cache()

distributed, local_rank, world_size, rank = init_distributed()
is_master = rank == 0

# Create output directories (only on master process)
if is_master:
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    print(f"Using {world_size} GPUs for training")

# Set the device
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

if distributed:
    # Ensure all processes are ready before continuing
    dist.barrier()
    print(f"Process {rank}/{world_size} ready on device {device}")

#######################
# Model Definitions
#######################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.channel_up = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.channel_up(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=128, n_downsampling=4):
        super().__init__()
        
        # Initial convolution layer
        layers = [
            nn.Conv2d(in_channels, hidden_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        current_dim = hidden_dims
        for i in range(n_downsampling):
            layers.append(ResidualBlock(current_dim, current_dim))
            layers.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_dim *= 2
        
        # Final convolution to get embeddings
        self.final_dim = current_dim
        layers.append(nn.Conv2d(current_dim, current_dim, kernel_size=3, padding=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dims=128, n_upsampling=4):
        super().__init__()
        
        # Calculate initial dimension based on upsampling
        current_dim = hidden_dims * (2 ** n_upsampling)
        
        # Initial convolution
        layers = [
            nn.Conv2d(current_dim, current_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        
        # Upsampling layers
        for i in range(n_upsampling):
            layers.append(ResidualBlock(current_dim, current_dim))
            layers.append(nn.ConvTranspose2d(current_dim, current_dim // 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_dim //= 2
            
        # Final convolution to get output image
        layers.append(nn.Conv2d(current_dim, out_channels, kernel_size=3, padding=1))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
    def forward(self, z):
        # z shape: [B, C, H, W]
        z = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_flattened = z.view(-1, self.embedding_dim)  # [B*H*W, C]
        
        # Calculate distances to codebook vectors
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.codebook.weight.t())
        
        # Find nearest codebook vector
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.codebook(min_encoding_indices).view(z.shape)
        
        # Compute loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.commitment_cost * torch.mean((z_q - z.detach()) ** 2)
        
        # Preserve gradients with straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Reshape back to match the input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, min_encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])


class FixedVQGAN(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=128, n_downsampling=4, codebook_size=1024, embedding_dim=256):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, n_downsampling)
        self.quantize = VectorQuantizer(codebook_size, embedding_dim)
        self.decoder = Decoder(in_channels, hidden_dims, n_downsampling)
        
        # Projection layer from encoder output dim to embedding dim
        self.proj_in = nn.Conv2d(self.encoder.final_dim, embedding_dim, kernel_size=1)
        
        # Projection layer from embedding dim to decoder input dim
        self.proj_out = nn.Conv2d(embedding_dim, self.encoder.final_dim, kernel_size=1)
        
    def encode(self, x):
        z = self.encoder(x)
        z = self.proj_in(z)
        z_q, _, indices = self.quantize(z)
        return z_q, indices
        
    def decode(self, z_q):
        # Apply proj_out if it's coming directly from the quantizer
        if z_q.size(1) == self.proj_out.in_channels:  # If channels match embedding_dim
            z_q = self.proj_out(z_q)
        # Otherwise, assume it's already been projected
        x_recon = self.decoder(z_q)
        return x_recon
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.proj_in(z)
        z_q, quantizer_loss, indices = self.quantize(z)
        x_recon = self.decode(z_q)
        
        return x_recon, quantizer_loss, indices
    
    def get_codebook_indices(self, x):
        z = self.encoder(x)
        z = self.proj_in(z)
        _, _, indices = self.quantize(z)
        return indices


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=64, n_layers=3):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, hidden_dims, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        current_dim = hidden_dims
        for i in range(1, n_layers):
            next_dim = min(current_dim * 2, 512)
            layers.append(DiscriminatorBlock(current_dim, next_dim))
            current_dim = next_dim
        
        # Final layer
        layers.append(nn.Conv2d(current_dim, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        
        # Get query, key, value projections
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Calculate attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, codebook_size, seq_len, dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.token_embedding = nn.Embedding(codebook_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)
        
        self.to_logits = nn.Linear(dim, codebook_size)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len]
        tokens = self.token_embedding(x)
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)
        
        # Apply transformer
        x = self.transformer(tokens)
        
        # Get logits
        logits = self.to_logits(x)
        
        return logits


class SuperResolutionModel(nn.Module):
    def __init__(self, vqgan, transformer, codebook_size, low_res_size=(32, 32), high_res_size=(256, 256)):
        super().__init__()
        self.vqgan = vqgan
        self.transformer = transformer
        self.codebook_size = codebook_size
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        
    def forward(self, x_low):
        """Forward pass for training"""
        # First upsample to high res to get the conditioning
        x_upscaled = F.interpolate(x_low, size=self.high_res_size, mode='bicubic', align_corners=False)
        
        # Get the indices from the VQGAN
        _, cond_indices = self.vqgan.encode(x_upscaled)
        
        # Flatten the spatial dimensions (flatten the indices)
        batch_size = x_low.shape[0]
        cond_indices_flat = cond_indices.view(batch_size, -1)
        
        # Create an input sequence for the transformer (we'll predict the next token)
        # For training, we use teacher forcing with the ground truth indices
        # For this example, we'll just use the same indices shifted
        input_indices = cond_indices_flat[:, :-1]
        target_indices = cond_indices_flat[:, 1:]
        
        # Forward through transformer to get logits
        logits = self.transformer(input_indices)
        
        return logits, target_indices
    
    def generate(self, x_low, temperature=1.0, top_k=None):
        """Generate high-resolution image from low-resolution input (simplified for testing)"""
        # First upsample to high res to get the conditioning
        x_upscaled = F.interpolate(x_low, size=self.high_res_size, mode='bicubic', align_corners=False)
        
        # Get the indices from the VQGAN
        _, cond_indices = self.vqgan.encode(x_upscaled)
        
        # Flatten the indices
        batch_size = x_low.shape[0]
        cond_indices_flat = cond_indices.view(batch_size, -1)
        
        # Create a sequence of the expected length (transformer.pos_embedding.shape[1]) filled with zeros
        expected_seq_len = self.transformer.pos_embedding.shape[1]
        sequence = torch.zeros((batch_size, expected_seq_len), dtype=torch.long, device=x_low.device)
        
        # Use the first tokens from our conditioning and complete with zeros
        tokens_to_use = min(expected_seq_len, cond_indices_flat.size(1))
        sequence[:, :tokens_to_use] = cond_indices_flat[:, :tokens_to_use]
        
        # Reshape to spatial form (16x16 grid = 256 tokens)
        side_len = int(np.sqrt(256))  # 16x16
        # Pad to 256 if needed
        if sequence.size(1) < 256:
            padding = torch.zeros((batch_size, 256 - sequence.size(1)), 
                                dtype=torch.long, device=x_low.device)
            sequence = torch.cat([sequence, padding], dim=1)
        # Or trim if larger
        else:
            sequence = sequence[:, :256]
            
        indices_spatial = sequence.reshape(batch_size, side_len, side_len)
        
        # Get codebook entries
        z_q = self.vqgan.quantize.codebook(indices_spatial.view(-1))
        z_q = z_q.view(batch_size, side_len, side_len, -1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Decode
        x_generated = self.vqgan.decode(z_q)
        
        return x_generated


class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, low_res_size=(32, 32), high_res_size=(256, 256)):
        """
        Dataset for super-resolution with separate high-res and low-res directories
        
        Args:
            hr_dir: Directory containing high-resolution images
            lr_dir: Directory containing low-resolution images
            low_res_size: Size of low-resolution images (for resizing if needed)
            high_res_size: Size of high-resolution images (for resizing if needed)
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        
        # Get image filenames (assuming matching filenames in both directories)
        self.image_filenames = sorted([f for f in os.listdir(hr_dir) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        
        # Transforms for high-resolution images
        self.transform_high = transforms.Compose([
            transforms.Resize(high_res_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Transforms for low-resolution images
        self.transform_low = transforms.Compose([
            transforms.Resize(low_res_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if is_master:
            print(f"Found {len(self.image_filenames)} image pairs")
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        
        # Load high-resolution image
        hr_path = os.path.join(self.hr_dir, img_filename)
        img_high = Image.open(hr_path).convert('RGB')
        img_high = self.transform_high(img_high)
        
        # Load low-resolution image
        lr_path = os.path.join(self.lr_dir, img_filename)
        img_low = Image.open(lr_path).convert('RGB')
        img_low = self.transform_low(img_low)
        
        return img_low, img_high


# Helper function for distributed training
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# Modified train_vqgan function with DDP support and resume capability
def train_vqgan(vqgan, discriminator, train_sampler, dataloader, num_epochs=100, lr=2e-4, checkpoint_dir=None, device='cuda', start_epoch=0):
    """Train the VQGAN model with DDP support"""
    if is_master:
        print(f"Training VQGAN for {num_epochs} epochs starting at epoch {start_epoch}...")
    
    # Clear CUDA cache before DDP
    torch.cuda.empty_cache()
    
    # Wrap model with DDP if distributed
    if distributed:
        vqgan = DDP(vqgan, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Setup optimizers
    opt_vq = torch.optim.Adam(vqgan.parameters(), lr=lr)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    # Setup learning rate scheduler for gradual decay
    scheduler_vq = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vq, T_max=num_epochs, eta_min=lr/10)
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=num_epochs, eta_min=lr/10)
    
    # If resuming, load optimizer and scheduler states
    if start_epoch > 0 and checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f"vqgan_checkpoint_{start_epoch}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            opt_vq.load_state_dict(checkpoint['optimizer_vq'])
            opt_disc.load_state_dict(checkpoint['optimizer_disc'])
            scheduler_vq.load_state_dict(checkpoint['scheduler_vq'])
            scheduler_disc.load_state_dict(checkpoint['scheduler_disc'])
            if is_master:
                print(f"Loaded optimizer and scheduler states from {checkpoint_path}")
    
    # Loss functions
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Perceptual loss (using a pretrained VGG)
    perceptual_loss = torchvision.models.vgg16(pretrained=True).features[:16].to(device).eval()
    for param in perceptual_loss.parameters():
        param.requires_grad = False
    
    def compute_perceptual(img1, img2):
        # Normalize to match VGG input
        img1 = F.interpolate(img1, size=(224, 224), mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=(224, 224), mode='bilinear', align_corners=False)
        
        feat1 = perceptual_loss(img1)
        feat2 = perceptual_loss(img2)
        return F.mse_loss(feat1, feat2)
    
    # Dictionary to track losses for plotting (only on master process)
    if is_master:
        # If resuming, load existing loss history
        if start_epoch > 0 and checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f"vqgan_checkpoint_{start_epoch}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                loss_history = checkpoint.get('loss_history', {
                    'rec_loss': [], 'vq_loss': [], 'perc_loss': [], 'gen_loss': [], 'disc_loss': []
                })
        else:
            loss_history = {
                'rec_loss': [], 'vq_loss': [], 'perc_loss': [], 'gen_loss': [], 'disc_loss': []
            }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler
        if distributed:
            train_sampler.set_epoch(epoch)
            
        epoch_start_time = time.time()
        total_rec_loss = 0
        total_vq_loss = 0
        total_perc_loss = 0
        total_gen_loss = 0
        total_disc_loss = 0
        
        for i, (_, imgs_high) in enumerate(dataloader):
            imgs_high = imgs_high.to(device)
            
            # Train Discriminator
            opt_disc.zero_grad()
            
            # Real images
            real_pred = discriminator(imgs_high)
            real_target = torch.ones_like(real_pred)
            loss_disc_real = bce_loss(real_pred, real_target)
            
            # Fake images
            with torch.no_grad():
                recon_imgs, _, _ = vqgan(imgs_high)
            fake_pred = discriminator(recon_imgs)
            fake_target = torch.zeros_like(fake_pred)
            loss_disc_fake = bce_loss(fake_pred, fake_target)
            
            # Total discriminator loss
            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc.backward()
            opt_disc.step()
            
            # Train Generator (VQGAN)
            opt_vq.zero_grad()
            
            # Reconstruction
            recon_imgs, vq_loss, _ = vqgan(imgs_high)
            rec_loss = l1_loss(recon_imgs, imgs_high)
            perc_loss = compute_perceptual(recon_imgs, imgs_high)
            
            # Generator adversarial loss
            fake_pred = discriminator(recon_imgs)
            gen_loss = bce_loss(fake_pred, torch.ones_like(fake_pred))
            
            # Total generator loss - adjusted weights for better balance
            lambda_rec = 1.0
            lambda_vq = 1.0  
            lambda_perc = 0.25  # Increased from 0.1
            lambda_adv = 0.1
            
            loss_gen = lambda_rec * rec_loss + lambda_vq * vq_loss + lambda_perc * perc_loss + lambda_adv * gen_loss
            loss_gen.backward()
            opt_vq.step()
            
            # Track losses
            total_rec_loss += rec_loss.item()
            total_vq_loss += vq_loss.item()
            total_perc_loss += perc_loss.item()
            total_gen_loss += gen_loss.item()
            total_disc_loss += loss_disc.item()
            
            # Free up memory
            torch.cuda.empty_cache()
            
            if is_master and i % 25 == 0:  # More frequent updates on master process
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"Rec: {rec_loss.item():.4f}, VQ: {vq_loss.item():.4f}, "
                      f"Perc: {perc_loss.item():.4f}, G: {gen_loss.item():.4f}, D: {loss_disc.item():.4f}")
                
        # Update learning rate schedulers
        scheduler_vq.step()
        scheduler_disc.step()
        
        # Compute distributed average losses
        if distributed:
            rec_avg = reduce_tensor(torch.tensor(total_rec_loss, device=device)).item() / len(dataloader)
            vq_avg = reduce_tensor(torch.tensor(total_vq_loss, device=device)).item() / len(dataloader)
            perc_avg = reduce_tensor(torch.tensor(total_perc_loss, device=device)).item() / len(dataloader)
            gen_avg = reduce_tensor(torch.tensor(total_gen_loss, device=device)).item() / len(dataloader)
            disc_avg = reduce_tensor(torch.tensor(total_disc_loss, device=device)).item() / len(dataloader)
        else:
            rec_avg = total_rec_loss / len(dataloader)
            vq_avg = total_vq_loss / len(dataloader)
            perc_avg = total_perc_loss / len(dataloader)
            gen_avg = total_gen_loss / len(dataloader)
            disc_avg = total_disc_loss / len(dataloader)
        
        # Save for plotting (only on master process)
        if is_master:
            loss_history['rec_loss'].append(rec_avg)
            loss_history['vq_loss'].append(vq_avg)
            loss_history['perc_loss'].append(perc_avg)
            loss_history['gen_loss'].append(gen_avg)
            loss_history['disc_loss'].append(disc_avg)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, "
                f"Rec: {rec_avg:.4f}, VQ: {vq_avg:.4f}, "
                f"Perc: {perc_avg:.4f}, G: {gen_avg:.4f}, D: {disc_avg:.4f}, "
                f"LR: {scheduler_vq.get_last_lr()[0]:.6f}")
            
            # Save some reconstructed images to check progress
            if (epoch + 1) % config['evaluate_every'] == 0 or epoch == 0:
                with torch.no_grad():
                    # Use unwrapped model for visualization
                    vqgan_to_vis = vqgan.module if distributed else vqgan
                    recon_imgs, _, _ = vqgan_to_vis(imgs_high[:4])
                    
                    # Convert images to grid
                    grid_original = make_grid(imgs_high[:4].cpu(), nrow=2, normalize=True)
                    grid_recon = make_grid(recon_imgs[:4].cpu(), nrow=2, normalize=True)
                    
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(grid_original.permute(1, 2, 0))
                    plt.title("Original")
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(grid_recon.permute(1, 2, 0))
                    plt.title("Reconstructed")
                    plt.axis('off')
                    
                    if checkpoint_dir:
                        plt.savefig(os.path.join(checkpoint_dir, f"vqgan_recon_epoch_{epoch+1}.png"))
                    plt.close()
                    
                    # Clear cache after visualization
                    torch.cuda.empty_cache()
                    
            # Save the model more frequently
            if checkpoint_dir and ((epoch + 1) % config['save_checkpoint_every'] == 0 or epoch == num_epochs - 1):
                checkpoint_path = os.path.join(checkpoint_dir, f"vqgan_checkpoint_{epoch+1}.pth")
                
                # Get state dict from unwrapped model for saving
                vqgan_state = vqgan.module.state_dict() if distributed else vqgan.state_dict()
                disc_state = discriminator.module.state_dict() if distributed else discriminator.state_dict()
                
                torch.save({
                    'vqgan': vqgan_state,
                    'discriminator': disc_state,
                    'epoch': epoch,
                    'optimizer_vq': opt_vq.state_dict(),
                    'optimizer_disc': opt_disc.state_dict(),
                    'scheduler_vq': scheduler_vq.state_dict(),
                    'scheduler_disc': scheduler_disc.state_dict(),
                    'loss_history': loss_history
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
    # Plot loss curves at the end of training (only on master process)
    if is_master and checkpoint_dir:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(loss_history['rec_loss'])
        plt.title('Reconstruction Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(loss_history['vq_loss'])
        plt.title('VQ Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(loss_history['gen_loss'])
        plt.title('Generator Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(loss_history['disc_loss'])
        plt.title('Discriminator Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "vqgan_training_losses.png"))
        plt.close()
    
    # Return unwrapped model
    if distributed:
        return vqgan.module
    else:
        return vqgan


# Modified train_transformer function with DDP support and resume capability
def train_transformer(super_res_model, train_sampler, dataloader, num_epochs=100, lr=1e-4, checkpoint_dir=None, device='cuda', start_epoch=0):
    """Train the transformer model for super-resolution with DDP support"""
    if is_master:
        print(f"Training transformer for {num_epochs} epochs starting at epoch {start_epoch}...")
    
    # Clear CUDA cache before DDP
    torch.cuda.empty_cache()
    
    vqgan = super_res_model.vqgan.to(device)
    transformer = super_res_model.transformer.to(device)
    
    # Freeze VQGAN parameters
    for param in vqgan.parameters():
        param.requires_grad = False
    
    # Wrap transformer with DDP if distributed
    if distributed:
        transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=is_master)
    
    # If resuming, load optimizer and scheduler states
    if start_epoch > 0 and checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f"transformer_checkpoint_{start_epoch}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if is_master:
                print(f"Loaded optimizer and scheduler states from {checkpoint_path}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # For early stopping
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    # Training history
    if is_master:
        # If resuming, load existing loss history
        if start_epoch > 0 and checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f"transformer_checkpoint_{start_epoch}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                history = checkpoint.get('loss_history', {'loss': []})
        else:
            history = {'loss': []}
    
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler
        if distributed:
            train_sampler.set_epoch(epoch)
            
        epoch_start_time = time.time()
        total_loss = 0
        
        for i, (imgs_low, imgs_high) in enumerate(dataloader):
            imgs_low = imgs_low.to(device)
            imgs_high = imgs_high.to(device)
            
            # Create a temporary model for forward pass
            # We need to be careful with forward pass in distributed mode
            if distributed:
                temp_model = SuperResolutionModel(
                    vqgan, 
                    transformer.module, 
                    config['codebook_size'],
                    low_res_size=(config['low_res'], config['low_res']),
                    high_res_size=(config['high_res'], config['high_res'])
                )
            else:
                temp_model = super_res_model
                
            # Forward pass
            logits, target_indices = temp_model(imgs_low)
            
            # Make sure shapes match for loss calculation
            min_len = min(logits.shape[0] * logits.shape[1], target_indices.shape[0] * target_indices.shape[1])
            flat_logits = logits.reshape(-1, config['codebook_size'])[:min_len]
            flat_targets = target_indices.reshape(-1)[:min_len]
            
            # Compute loss
            loss = criterion(flat_logits, flat_targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Free up memory
            torch.cuda.empty_cache()
            
            if is_master and i % 25 == 0:  # More frequent updates on master process
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # Compute distributed average loss
        if distributed:
            avg_loss = reduce_tensor(torch.tensor(total_loss, device=device)).item() / len(dataloader)
        else:
            avg_loss = total_loss / len(dataloader)
        
        if is_master:
            history['loss'].append(avg_loss)
            epoch_time = time.time() - epoch_start_time
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                if checkpoint_dir:
                    best_model_path = os.path.join(checkpoint_dir, "transformer_best.pth")
                    
                    # Get state dict from unwrapped model for saving
                    transformer_state = transformer.module.state_dict() if distributed else transformer.state_dict()
                    
                    torch.save({
                        'transformer': transformer_state,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss
                    }, best_model_path)
                    print(f"Saved best model with loss {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Generate some super-resolution images to check progress
            if (epoch + 1) % config['evaluate_every'] == 0 or epoch == 0:
                with torch.no_grad():
                    try:
                        # Create temporary model for generation
                        if distributed:
                            temp_model = SuperResolutionModel(
                                vqgan, 
                                transformer.module, 
                                config['codebook_size'],
                                low_res_size=(config['low_res'], config['low_res']),
                                high_res_size=(config['high_res'], config['high_res'])
                            )
                        else:
                            temp_model = super_res_model
                            
                        # Generate high-res images from low-res inputs
                        generated_imgs = temp_model.generate(imgs_low[:2], temperature=0.8, top_k=100)
                        
                        # Convert images to grid
                        grid_low = make_grid(torch.clamp(imgs_low[:2] * 0.5 + 0.5, 0, 1).cpu(), nrow=2, normalize=False)
                        grid_high = make_grid(torch.clamp(imgs_high[:2] * 0.5 + 0.5, 0, 1).cpu(), nrow=2, normalize=False)
                        grid_gen = make_grid(torch.clamp(generated_imgs[:2] * 0.5 + 0.5, 0, 1).cpu(), nrow=2, normalize=False)
                        
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 3, 1)
                        plt.imshow(grid_low.permute(1, 2, 0))
                        plt.title("Low Resolution")
                        plt.axis('off')
                        plt.subplot(1, 3, 2)
                        plt.imshow(grid_gen.permute(1, 2, 0))
                        plt.title("Generated High Resolution")
                        plt.axis('off')
                        plt.subplot(1, 3, 3)
                        plt.imshow(grid_high.permute(1, 2, 0))
                        plt.title("Ground Truth High Resolution")
                        plt.axis('off')
                        
                        if checkpoint_dir:
                            plt.savefig(os.path.join(checkpoint_dir, f"superres_epoch_{epoch+1}.png"))
                        plt.close()
                        
                        # Clear cache after visualization
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error generating samples: {e}")
            
            # Regular checkpoint saving
            if checkpoint_dir and ((epoch + 1) % config['save_checkpoint_every'] == 0 or epoch == num_epochs - 1):
                checkpoint_path = os.path.join(checkpoint_dir, f"transformer_checkpoint_{epoch+1}.pth")
                
                # Get state dict from unwrapped model for saving
                transformer_state = transformer.module.state_dict() if distributed else transformer.state_dict()
                
                torch.save({
                    'transformer': transformer_state,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss_history': history
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
    
    # Plot training loss (only on master process)
    if is_master and checkpoint_dir:
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'])
        plt.title('Transformer Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(checkpoint_dir, "transformer_training_loss.png"))
        plt.close()
            
    # Load the best model before returning
    if is_master and os.path.exists(os.path.join(checkpoint_dir, "transformer_best.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_dir, "transformer_best.pth"))
        
        # Load state dict to unwrapped model
        if distributed:
            transformer.module.load_state_dict(checkpoint['transformer'])
        else:
            transformer.load_state_dict(checkpoint['transformer'])
            
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.4f}")
    
    # Synchronize processes to make sure all have loaded the best model
    if distributed:
        dist.barrier()
            
    # Return the unwrapped transformer model
    if distributed:
        super_res_model.transformer = transformer.module
    else:
        super_res_model.transformer = transformer
            
    return super_res_model


def test_super_resolution(super_res_model, dataloader, output_dir=None, device='cuda'):
    """Test the super-resolution model with basic metrics"""
    if is_master:
        print("Evaluating super-resolution model...")
    
    super_res_model = super_res_model.to(device)
    super_res_model.eval()
    
    # Use a simpler MSE/PSNR calculation
    def calculate_psnr(img1, img2):
        mse = ((img1 - img2) ** 2).mean()
        if mse == 0:
            return 100
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Metrics
    psnr_values = []
    
    with torch.no_grad():
        for i, (imgs_low, imgs_high) in enumerate(dataloader):
            if i >= 10:  # Only evaluate on a subset for efficiency
                break
                
            imgs_low = imgs_low.to(device)
            imgs_high = imgs_high.to(device)
            
            # Generate high-res images
            generated_imgs = super_res_model.generate(imgs_low, temperature=0.8, top_k=100)
            
            # Convert to numpy for metric calculation
            imgs_high_np = torch.clamp(imgs_high * 0.5 + 0.5, 0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            generated_np = torch.clamp(generated_imgs * 0.5 + 0.5, 0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            
            # Calculate metrics
            for j in range(len(imgs_high_np)):
                # PSNR
                psnr_val = calculate_psnr(imgs_high_np[j], generated_np[j])
                psnr_values.append(psnr_val)
            
            # Visualize samples (only on master process)
            if is_master and i < 5 and output_dir:
                grid_low = make_grid(torch.clamp(imgs_low * 0.5 + 0.5, 0, 1).cpu(), nrow=2, normalize=False)
                grid_gen = make_grid(torch.clamp(generated_imgs * 0.5 + 0.5, 0, 1).cpu(), nrow=2, normalize=False)
                grid_high = make_grid(torch.clamp(imgs_high * 0.5 + 0.5, 0, 1).cpu(), nrow=2, normalize=False)
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(grid_low.permute(1, 2, 0))
                plt.title("Low Resolution")
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(grid_gen.permute(1, 2, 0))
                plt.title(f"Generated (PSNR: {psnr_values[-1]:.2f}dB)")
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(grid_high.permute(1, 2, 0))
                plt.title("Ground Truth")
                plt.axis('off')
                
                plt.savefig(os.path.join(output_dir, f"test_sample_{i+1}.png"))
                plt.close()
                
                # Clear cache after visualization
                torch.cuda.empty_cache()
    
    # Gather and average PSNR values from all processes
    if distributed:
        # Convert local values to tensor
        local_psnr = torch.tensor(psnr_values, device=device,dtype=torch.float32)
        
        # Get size from each process
        local_size = torch.tensor([len(psnr_values)], device=device)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        
        # Create list of receiving tensors with appropriate size
        max_size = max([size.item() for size in gathered_sizes])
        if max_size > 0:  # Only if at least one process has values
            gathered_psnr = [torch.zeros(max_size, device=device) for _ in range(world_size)]
            
            # Pad local tensor if needed
            if len(local_psnr) < max_size:
                local_psnr = torch.cat([local_psnr, torch.zeros(max_size - len(local_psnr), device=device)])
            
            # Gather all PSNR values
            dist.all_gather(gathered_psnr, local_psnr)
            
            # Combine values (skipping padding zeros)
            all_psnr = []
            for i, size in enumerate(gathered_sizes):
                if size.item() > 0:
                    all_psnr.extend(gathered_psnr[i][:size.item()].cpu().numpy())
                
            psnr_values = all_psnr
    
    # Calculate and print average metrics (only master process)
    if is_master:
        if psnr_values:
            avg_psnr = np.mean(psnr_values)
            
            print(f"Evaluation completed:")
            print(f"Average PSNR: {avg_psnr:.2f} dB")
            
            # Save metrics to file
            if output_dir:
                with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
                    f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        else:
            print("No evaluation data available")
    
    return psnr_values


def find_latest_checkpoint(checkpoint_dir, prefix):
    """Find the latest checkpoint file with the given prefix"""
    if not os.path.exists(checkpoint_dir):
        return None, 0
        
    latest_epoch = -1
    latest_checkpoint = None
    
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(prefix) and filename.endswith('.pth'):
            try:
                epoch_str = filename.split('_')[-1].split('.')[0]
                epoch = int(epoch_str)
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = os.path.join(checkpoint_dir, filename)
            except (ValueError, IndexError):
                continue
                
    return latest_checkpoint, latest_epoch


def main():
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    # Create dataset and dataloaders
    low_res_size = (config['low_res'], config['low_res'])
    high_res_size = (config['high_res'], config['high_res'])

    dataset = SuperResolutionDataset(config['hr_dir'], config['lr_dir'], 
                                   low_res_size=low_res_size, 
                                   high_res_size=high_res_size)

    # Create train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True  # Important for DDP to have same batch size on all ranks
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        sampler=test_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Calculate sequence length for transformer
    seq_len = 255  # As determined in our testing

    # Create models with reduced dimensions
    n_downsampling = 4  # Fixed parameter
    vqgan = FixedVQGAN(
        in_channels=3, 
        hidden_dims=config['hidden_dims'],
        n_downsampling=n_downsampling,
        codebook_size=config['codebook_size'], 
        embedding_dim=config['embedding_dim']
    ).to(device)

    discriminator = PatchDiscriminator(
        in_channels=3,
        hidden_dims=64,
        n_layers=3
    ).to(device)

    transformer = TransformerModel(
        codebook_size=config['codebook_size'],
        seq_len=seq_len,
        dim=config['transformer_dim'],
        depth=config['transformer_depth'],
        heads=config['transformer_heads'],
        mlp_dim=config['transformer_dim'] * 4,
        dropout=config['transformer_dropout']
    ).to(device)

    super_res_model = SuperResolutionModel(
        vqgan, 
        transformer, 
        config['codebook_size'],
        low_res_size=low_res_size,
        high_res_size=high_res_size
    )

    # Check for existing checkpoints if resuming
    vqgan_start_epoch = 0
    transformer_start_epoch = 0
    
    if args.resume:
        # Look for VQGAN checkpoint
        vqgan_checkpoint, vqgan_epoch = find_latest_checkpoint(config['checkpoint_dir'], 'vqgan_checkpoint_')
        if vqgan_checkpoint and is_master:
            print(f"Found VQGAN checkpoint at epoch {vqgan_epoch}")
            checkpoint = torch.load(vqgan_checkpoint, map_location=device)
            vqgan.load_state_dict(checkpoint['vqgan'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            vqgan_start_epoch = vqgan_epoch
        
        # Look for transformer checkpoint
        transformer_checkpoint, transformer_epoch = find_latest_checkpoint(config['checkpoint_dir'], 'transformer_checkpoint_')
        if transformer_checkpoint and is_master:
            print(f"Found transformer checkpoint at epoch {transformer_epoch}")
            checkpoint = torch.load(transformer_checkpoint, map_location=device)
            transformer.load_state_dict(checkpoint['transformer'])
            transformer_start_epoch = transformer_epoch

    # Train VQGAN if needed
    if vqgan_start_epoch < config['vqgan_epochs']:
        if is_master:
            print(f"Training VQGAN from epoch {vqgan_start_epoch}...")
        vqgan = train_vqgan(vqgan, discriminator, train_sampler, train_dataloader, 
                          num_epochs=config['vqgan_epochs'], lr=config['vqgan_lr'], 
                          checkpoint_dir=config['checkpoint_dir'], device=device,
                          start_epoch=vqgan_start_epoch)
    else:
        if is_master:
            print("VQGAN training already completed, skipping to transformer training")

    # Update the super_res_model with the trained VQGAN
    super_res_model.vqgan = vqgan

    # Train Transformer if needed
    if transformer_start_epoch < config['transformer_epochs']:
        if is_master:
            print(f"Training transformer from epoch {transformer_start_epoch}...")
        super_res_model = train_transformer(super_res_model, train_sampler, train_dataloader, 
                                          num_epochs=config['transformer_epochs'], lr=config['transformer_lr'],
                                          checkpoint_dir=config['checkpoint_dir'], device=device,
                                          start_epoch=transformer_start_epoch)
    else:
        if is_master:
            print("Transformer training already completed, skipping to evaluation")

    # Evaluate the model
    if is_master:
        print("Evaluating model...")
    test_super_resolution(super_res_model, test_dataloader, 
                        output_dir=config['output_dir'], device=device)


if __name__ == "__main__":
    main()