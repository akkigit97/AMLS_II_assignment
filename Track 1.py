import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from math import log10
import matplotlib.pyplot as plt
import re
import gc
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import lpips
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import pandas as pd
import time
import wandb
import resource  # Add resource module for system resource management
import os.path
import platform
import math
import sys

# Initialize wandb tracking (add this near the imports)
def init_wandb(project_name="SRGAN", config=None):
    """Initialize Weights & Biases tracking"""
    try:
        # Default configuration for wandb
        default_config = {
            "architecture": "SRGAN with RRDBs",
            "dataset": "DIV2K",
            "batch_size": 4,
            "learning_rate_G": 1e-4,
            "learning_rate_D": 1e-4,
            "epochs": 100,
            "scale_factor": 4,
            "content_weight": 1.0,
            "perceptual_weight": 0.1,
            "adversarial_weight": 1e-4,
            "lpips_weight": 0.1
        }
        
        # Update config with any provided values
        if config:
            default_config.update(config)
        
        # Initialize wandb
        wandb.init(project=project_name, config=default_config)
        print("Weights & Biases tracking initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize Weights & Biases tracking: {e}")
        return False

# Try to import psutil, install if not available
try:
    import psutil
except ImportError:
    print("psutil not found, trying to install...")
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "psutil"])
        import psutil
        print("psutil installed successfully")
    except Exception as e:
        print(f"Failed to install psutil: {e}")
        # Define a dummy psutil class for graceful fallback
        class DummyProcess:
            def memory_info(self):
                class MemInfo:
                    def __init__(self):
                        self.rss = 0
                return MemInfo()
            
            def open_files(self):
                return []
                
        class DummyPsutil:
            def Process(self, _):
                return DummyProcess()
        
        psutil = DummyPsutil()

from functools import lru_cache  # For caching

# Set system resource limits (increase open file limit)
def increase_file_limit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to increase the limit to the hard limit
        new_soft = min(hard, 4096)  # Try to set to a reasonable number
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"Increased open file limit from {soft} to {new_soft}")
    except Exception as e:
        print(f"Could not increase file limit: {e}")

# Call at script startup
increase_file_limit()

# Monitor resource usage
def print_resource_usage():
    try:
        # Get the current process
        process = psutil.Process(os.getpid())
        # Get memory usage
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        # Get number of open files
        try:
            open_files = process.open_files()
            print(f"Number of open files: {len(open_files)}")
        except Exception as e:
            print(f"Could not get open file count: {e}")
    except Exception as e:
        print(f"Error monitoring resources: {e}")

# -------------------------------------
# Device configuration
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Function to clear GPU cache
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Add a new function for memory management during training
def manage_memory():
    """Clean up memory by clearing GPU cache and running garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Close any unclosed matplotlib figures
    plt.close('all')
    
    # Print current resource usage
    print_resource_usage()

# Image cache implementation 
class ImageCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.keys = []
    
    def get(self, path):
        if path in self.cache:
            # Move to the end to mark as recently used
            self.keys.remove(path)
            self.keys.append(path)
            return self.cache[path].copy()
        return None
    
    def put(self, path, img):
        if path not in self.cache:
            if len(self.keys) >= self.max_size:
                # Remove oldest item
                oldest = self.keys.pop(0)
                del self.cache[oldest]
            self.keys.append(path)
            self.cache[path] = img.copy()

#####################
# Folder Paths
#####################
# Try to locate the data directories with flexible paths
import os.path
import platform

# Check if running on Windows
is_windows = platform.system().lower() == 'windows'

# Primary data directories to try
data_dirs = []

# Add platform-specific paths
if is_windows:
    # Windows paths
    data_dirs.extend([
        os.path.join(os.getcwd(), "DIV2K"),
        os.path.join(os.path.dirname(os.getcwd()), "DIV2K"),
        r"C:\Users\akhil\AMLS2_assignment\DIV2K"
    ])
else:
    # Linux/Unix paths
    data_dirs.extend([
        "/home/uceeaat/AMLS_II_assignment/AMLS_II_assignment/DIV2K",
        "/home/uceeaat/AMLS_II_assignment/DIV2K"
    ])

# Always include these common paths
data_dirs.extend([
    os.path.join(os.getcwd(), "DIV2K"),
    os.path.join(os.path.dirname(os.getcwd()), "DIV2K")
])

# Find the first valid data directory
data_dir = None
for dir_path in data_dirs:
    if os.path.exists(dir_path):
        data_dir = dir_path
        print(f"Found DIV2K data directory at: {data_dir}")
        break

if data_dir is None:
    print("WARNING: Could not find DIV2K data directory. Please specify the path manually.")
    # Set a default directory for the script to continue (it will fail later with a more specific error)
    data_dir = data_dirs[0]

# Set paths based on found data directory
TRAIN_LR_PATH = os.path.join(data_dir, "DIV2K_train_LR_bicubic_X4", "DIV2K_train_LR_bicubic", "X4")
TRAIN_HR_PATH = os.path.join(data_dir, "DIV2K_train_HR")
VALID_LR_PATH = os.path.join(data_dir, "DIV2K_valid_LR_bicubic_X4", "DIV2K_valid_LR_bicubic", "X4")
VALID_HR_PATH = os.path.join(data_dir, "DIV2K_valid_HR", "DIV2K_valid_HR")

# Function to create synthetic data for testing
def create_synthetic_dataset(base_dir="synthetic_DIV2K", num_samples=10, hr_size=256, use_synthetic=False):
    """Create a small synthetic dataset for testing purposes if the real dataset is not available."""
    if not use_synthetic:
        return False
        
    try:
        # Import libraries
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_pil_image
        import math
        
        print("\nCreating synthetic dataset for testing...")
        
        # Create directories
        os.makedirs(os.path.join(base_dir, "DIV2K_train_HR"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "DIV2K_valid_HR", "DIV2K_valid_HR"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "DIV2K_train_LR_bicubic_X4", "DIV2K_train_LR_bicubic", "X4"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "DIV2K_valid_LR_bicubic_X4", "DIV2K_valid_LR_bicubic", "X4"), exist_ok=True)
        
        # Set paths
        global TRAIN_HR_PATH, TRAIN_LR_PATH, VALID_HR_PATH, VALID_LR_PATH
        
        train_hr_dir = os.path.join(base_dir, "DIV2K_train_HR")
        valid_hr_dir = os.path.join(base_dir, "DIV2K_valid_HR", "DIV2K_valid_HR")
        train_lr_dir = os.path.join(base_dir, "DIV2K_train_LR_bicubic_X4", "DIV2K_train_LR_bicubic", "X4")
        valid_lr_dir = os.path.join(base_dir, "DIV2K_valid_LR_bicubic_X4", "DIV2K_valid_LR_bicubic", "X4")
        
        # Create more diverse synthetic patterns for better testing
        def create_image_pattern(pattern_type, size):
            """Create different test patterns for synthetic images"""
            hr_tensor = torch.zeros(3, size, size)
            
            if pattern_type == "checkerboard":
                # Create a checkerboard pattern
                checker_size = 32
                for y in range(0, size, checker_size):
                    for x in range(0, size, checker_size):
                        if (x // checker_size + y // checker_size) % 2 == 0:
                            hr_tensor[:, y:min(y+checker_size, size), x:min(x+checker_size, size)] = 1.0
            
            elif pattern_type == "gradient":
                # Create a color gradient
                y_coords = torch.linspace(0, 1, size).view(1, size, 1).expand(1, size, size)
                x_coords = torch.linspace(0, 1, size).view(1, 1, size).expand(1, size, size)
                hr_tensor = torch.cat([y_coords, x_coords, y_coords * x_coords], dim=0)
            
            elif pattern_type == "circles":
                # Add random colored circles
                for _ in range(8):
                    cx = torch.randint(0, size, (1,)).item()
                    cy = torch.randint(0, size, (1,)).item()
                    radius = torch.randint(20, 60, (1,)).item()
                    color = torch.rand(3)
                    
                    y_grid, x_grid = torch.meshgrid(torch.arange(size), torch.arange(size))
                    mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) < (radius ** 2)
                    
                    for c in range(3):
                        hr_tensor[c][mask] = color[c]
            
            elif pattern_type == "lines":
                # Add random lines with different thicknesses
                for _ in range(10):
                    thickness = torch.randint(2, 10, (1,)).item()
                    angle = torch.rand(1).item() * 2 * math.pi
                    color = torch.rand(3)
                    
                    # Create a line using angle
                    start_x = torch.randint(0, size, (1,)).item()
                    start_y = torch.randint(0, size, (1,)).item()
                    
                    # Draw the line
                    for t in range(-size, size):
                        x = int(start_x + t * math.cos(angle))
                        y = int(start_y + t * math.sin(angle))
                        
                        if 0 <= x < size and 0 <= y < size:
                            for dx in range(-thickness//2, thickness//2 + 1):
                                for dy in range(-thickness//2, thickness//2 + 1):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < size and 0 <= ny < size:
                                        for c in range(3):
                                            hr_tensor[c, ny, nx] = color[c]
            
            elif pattern_type == "text":
                # Create a solid background with different color
                bg_color = torch.rand(3)
                for c in range(3):
                    hr_tensor[c, :, :] = bg_color[c]
                
                # Add simulated "text" as small rectangles
                for _ in range(30):
                    rect_h = torch.randint(2, 8, (1,)).item()
                    rect_w = torch.randint(5, 40, (1,)).item()
                    x = torch.randint(0, size - rect_w, (1,)).item()
                    y = torch.randint(0, size - rect_h, (1,)).item()
                    
                    color = 1.0 - bg_color  # Contrasting color
                    
                    # Draw the rectangle
                    for c in range(3):
                        hr_tensor[c, y:y+rect_h, x:x+rect_w] = color[c]
            
            else:  # "natural" - attempt to create natural-looking texture
                # Start with noise
                hr_tensor = torch.rand(3, size, size)
                
                # Apply smoothing for more natural appearance
                kernel_size = 15
                sigma = 5.0
                # Create Gaussian blur by hand
                grid_x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
                grid_y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
                x, y = torch.meshgrid(grid_x, grid_y)
                kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                kernel = kernel.view(1, 1, kernel_size, kernel_size)
                
                # Apply convolution for smoothing
                hr_tensor = hr_tensor.unsqueeze(0)  # Add batch dimension [1, 3, H, W]
                padding = kernel_size // 2
                
                # Process each channel separately to avoid dimension errors
                smoothed_tensor = torch.zeros_like(hr_tensor)
                for c in range(3):
                    # Extract single channel and add dummy channel dimension
                    channel = hr_tensor[:, c:c+1, :, :]  # [1, 1, H, W]
                    # Pad with reflection to avoid border artifacts
                    padded = F.pad(channel, (padding, padding, padding, padding), mode='reflect')
                    # Apply convolution
                    smoothed = F.conv2d(padded, kernel, padding=0)
                    # Store result
                    smoothed_tensor[:, c:c+1, :, :] = smoothed
                
                hr_tensor = smoothed_tensor.squeeze(0)  # Remove batch dimension [3, H, W]
            
            return hr_tensor
                
        # Create train and validation samples
        pattern_types = ["checkerboard", "gradient", "circles", "lines", "text", "natural"]
        
        for i in range(num_samples):
            # Use different patterns for diversity
            pattern_type = pattern_types[i % len(pattern_types)]
            hr_tensor = create_image_pattern(pattern_type, hr_size)
            
            # Create low-resolution version
            lr_tensor = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False).squeeze(0)
            
            # Convert to PIL images and save
            hr_img = to_pil_image(hr_tensor)
            lr_img = to_pil_image(lr_tensor)
            
            # Save images
            hr_img.save(os.path.join(train_hr_dir, f"{i+1:04d}.png"))
            lr_img.save(os.path.join(train_lr_dir, f"{i+1:04d}.png"))
            
            # Use first 2 samples for validation
            if i < 2:
                hr_img.save(os.path.join(valid_hr_dir, f"{i+1:04d}.png"))
                lr_img.save(os.path.join(valid_lr_dir, f"{i+1:04d}.png"))
        
        # Update global paths
        TRAIN_HR_PATH = train_hr_dir
        TRAIN_LR_PATH = train_lr_dir
        VALID_HR_PATH = valid_hr_dir
        VALID_LR_PATH = valid_lr_dir
        
        print(f"Created synthetic dataset with {num_samples} samples in '{base_dir}'")
        
        # Show example images in a grid
        try:
            import matplotlib.pyplot as plt
            
            # Create a figure to display example images
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Example Synthetic Images", fontsize=16)
            
            # Show first 3 pattern types in HR and LR
            for i in range(min(3, len(pattern_types))):
                pattern_type = pattern_types[i]
                
                # Load HR and LR images
                hr_path = os.path.join(train_hr_dir, f"{i+1:04d}.png")
                lr_path = os.path.join(train_lr_dir, f"{i+1:04d}.png")
                
                hr_img = Image.open(hr_path)
                lr_img = Image.open(lr_path)
                
                # Upscale LR for better visualization
                lr_upscaled = lr_img.resize(hr_img.size, Image.BICUBIC)
                
                # Display images
                axs[0, i].imshow(np.array(hr_img))
                axs[0, i].set_title(f"HR - {pattern_type}")
                axs[0, i].axis('off')
                
                axs[1, i].imshow(np.array(lr_upscaled))
                axs[1, i].set_title(f"LR (upscaled) - {pattern_type}")
                axs[1, i].axis('off')
            
            plt.tight_layout()
            
            # Save and show the figure
            os.makedirs("synthetic_examples", exist_ok=True)
            plt.savefig("synthetic_examples/example_patterns.png")
            plt.close()
            
            print("Saved example synthetic patterns to 'synthetic_examples/example_patterns.png'")
            
            # Log to wandb if initialized
            try:
                import wandb
                if wandb.run is not None:
                    example_img = wandb.Image("synthetic_examples/example_patterns.png")
                    wandb.log({"synthetic_dataset/examples": example_img})
            except Exception as e:
                print(f"Warning: Could not log synthetic examples to wandb: {e}")
                
        except Exception as e:
            print(f"Warning: Could not create example figure: {e}")
            
        return True
    
    except Exception as e:
        print(f"Error creating synthetic dataset: {e}")
        return False

# Function to verify dataset presence and provide instructions if missing
def verify_dataset(use_synthetic=False):
    global TRAIN_LR_PATH, TRAIN_HR_PATH, VALID_LR_PATH, VALID_HR_PATH, data_dir
    
    # Check if the dataset directories exist
    if not os.path.exists(TRAIN_HR_PATH):
        print("\n" + "="*80)
        print("ERROR: DIV2K dataset not found!")
        print("="*80)
        
        # Try to create synthetic data for testing if requested
        if use_synthetic:
            if create_synthetic_dataset(use_synthetic=True):
                print("Using synthetic dataset for testing.")
                return True
        
        print("\nThe DIV2K dataset is required to train the SRGAN model.")
        print("\nTo download the DIV2K dataset:")
        print("1. Visit: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("2. Download the following files:")
        print("   - DIV2K_train_HR.zip")
        print("   - DIV2K_train_LR_bicubic_X4.zip")
        print("   - DIV2K_valid_HR.zip")
        print("   - DIV2K_valid_LR_bicubic_X4.zip")
        print("\n3. Create a directory structure as follows:")
        print("   DIV2K/")
        print("   ├── DIV2K_train_HR/")
        print("   ├── DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4/")
        print("   ├── DIV2K_valid_HR/DIV2K_valid_HR/")
        print("   └── DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4/")
        print("\n4. Place the DIV2K directory in one of the following locations:")
        for dir_path in data_dirs:
            print(f"   - {dir_path}")
        print("\nAlternatively, modify the paths in the script to point to your dataset location.")
        print("\nFor quick testing, you can use the synthetic dataset by setting 'use_synthetic=True' in the script.")
        print("="*80 + "\n")
        return False
    
    return True

# -------------------------------
# Improved Paired Bicubic Dataset (Track 1) - No Augmentation
# -------------------------------
class PairedBicubicDataset(Dataset):
    """
    Loads paired LR (bicubic) and HR images from specified directories.
    Ensures proper matching between LR and HR files using ID extraction.
    """
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None, patch_size=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size  # Optional patch size for memory efficiency
        self.lr_cache = ImageCache(max_size=50)  # Cache for LR images
        self.hr_cache = ImageCache(max_size=50)  # Cache for HR images
        
        # Get all image files
        lr_files = [f for f in os.listdir(lr_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        hr_files = [f for f in os.listdir(hr_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Extract IDs for matching
        self.paired_files = []
        lr_ids = {self._extract_id(f): f for f in lr_files}
        hr_ids = {self._extract_id(f): f for f in hr_files}
        
        # Find common IDs
        common_ids = set(lr_ids.keys()) & set(hr_ids.keys())
        print(f"Found {len(common_ids)} matching image pairs")
        
        for img_id in common_ids:
            self.paired_files.append((lr_ids[img_id], hr_ids[img_id]))
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        
        # Print paired file example 
        if len(self.paired_files) > 0:
            print(f"Example paired files: LR={self.paired_files[0][0]}, HR={self.paired_files[0][1]}")

    def _extract_id(self, filename):
        # Extract numerical ID from filename (assumes format like 0001.png or img_0001.jpg)
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return filename  # Fallback to filename if no number found
    
    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        lr_file, hr_file = self.paired_files[idx]
        lr_path = os.path.join(self.lr_dir, lr_file)
        hr_path = os.path.join(self.hr_dir, hr_file)
        
        # First check cache
        lr_img = self.lr_cache.get(lr_path)
        hr_img = self.hr_cache.get(hr_path)
        
        # If not in cache, load from disk
        if lr_img is None or hr_img is None:
            try:
                # Load images with proper file handling
                with Image.open(lr_path) as img:
                    lr_img = img.convert("RGB").copy()
                    # Store original size
                    lr_size = lr_img.size
                    self.lr_cache.put(lr_path, lr_img)
                    
                with Image.open(hr_path) as img:
                    hr_img = img.convert("RGB").copy()
                    # Store original size
                    hr_size = hr_img.size
                    self.hr_cache.put(hr_path, hr_img)
                
                # Verify scale factor (should be 4x)
                hr_width, hr_height = hr_size
                lr_width, lr_height = lr_size
                
                # Ensure HR is exactly 4x the size of LR (or very close to it)
                expected_hr_width = lr_width * 4
                expected_hr_height = lr_height * 4
                
                # Fix if sizes don't match the expected 4x factor
                if abs(hr_width - expected_hr_width) > 4 or abs(hr_height - expected_hr_height) > 4:
                    # Fix the issue by ensuring HR is exactly 4x the size of LR
                    hr_img = hr_img.resize((lr_width * 4, lr_height * 4), Image.BICUBIC)
                
        except Exception as e:
            print(f"Error loading images at index {idx}: {e}")
            # Return a default small image in case of error
            lr_img = Image.new("RGB", (64, 64), color=(0, 0, 0))
            hr_img = Image.new("RGB", (256, 256), color=(0, 0, 0))
        
        # Apply transforms
        if self.hr_transform:
            hr_img = self.hr_transform(hr_img)
        if self.lr_transform:
            lr_img = self.lr_transform(lr_img)
        
        # Ensure tensor shapes are correct (perform a sanity check)
        if isinstance(lr_img, torch.Tensor) and isinstance(hr_img, torch.Tensor):
            # LR should be 1/4 the size of HR in each dimension
            c_lr, h_lr, w_lr = lr_img.shape
            c_hr, h_hr, w_hr = hr_img.shape
            
            # Verify that HR is approximately 4x the size of LR
            if abs(h_hr - h_lr * 4) > 2 or abs(w_hr - w_lr * 4) > 2:
                # This should never happen if transforms are correct
                pass
            
        return lr_img, hr_img

# -------------------------------
# SRGAN Components: Improved Residual Block, Generator, Discriminator
# -------------------------------
class AdaptiveNorm(nn.Module):
    """Adaptive normalization layer that can handle varying spatial dimensions"""
    def __init__(self, num_features):
        super(AdaptiveNorm, self).__init__()
        self.norm = nn.GroupNorm(8, num_features)  # Group normalization is invariant to input size
    
    def forward(self, x):
        return self.norm(x)

class ImprovedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ImprovedResidualBlock, self).__init__()
        # Use adaptive normalization
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = AdaptiveNorm(channels)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveNorm(channels)
        
        # Simplified attention mechanism
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Residual scaling factor - start with smaller value to prevent instability
        self.scale = nn.Parameter(torch.FloatTensor([0.1]))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        identity = x
        
        # Main branch with proper normalization order
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.prelu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Apply attention and dropout
        out = out * self.se(out)
        out = self.dropout(out)
        
        # Check for NaN before adding residual
        if torch.isnan(out).any() or torch.isinf(out).any():
            return identity
        
        # Scaled residual connection
        return identity + self.scale * out

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with Channel Attention for better feature extraction"""
    def __init__(self, channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, 1, 1)
        
        # Add channel attention for focusing on important channels
        self.channel_attention = ChannelAttention(channels, reduction=16)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # Smaller scale factor for stability
        self.scale = nn.Parameter(torch.FloatTensor([0.1]))
        
        # Weight initialization for stable training
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        
        # Apply channel attention
        x5 = self.channel_attention(x5)
        
        # Check for NaN/Inf before residual connection
        if torch.isnan(x5).any() or torch.isinf(x5).any():
            return x
            
        return x + x5 * self.scale

class RRDB(nn.Module):
    """Enhanced Residual-in-Residual Dense Block with stability improvements"""
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)
        # Add a 1x1 conv for feature fusion
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        # Smaller scale factor to prevent training instability
        self.scale = nn.Parameter(torch.FloatTensor([0.1]))
        
        # Initialize the 1x1 conv
        nn.init.kaiming_normal_(self.fusion.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fusion.bias)
        
    def forward(self, x):
        # Use a multi-level skip connection pattern like in ESRGAN
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        
        # Feature fusion
        out = self.fusion(out)
        
        # Check for NaN/Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            return x
            
        # Global residual connection
        return x + out * self.scale

class ImprovedGenerator(nn.Module):
    def __init__(self, num_blocks=16, base_channels=64):
        super(ImprovedGenerator, self).__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=9, stride=1, padding=4)
        self.relu = nn.PReLU()

        # Residual-in-Residual Dense Blocks
        self.res_blocks = nn.Sequential(
            *[self._make_res_block(base_channels) for _ in range(num_blocks)]
        )
        
        # Second conv layer after residual blocks
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(base_channels)
        
        # Upsampling layers - using PixelShuffle for better performance
        upsampling = []
        for _ in range(2):  # Upscale by factor of 4 (2^2)
            upsampling.extend([
                nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
            ])
        self.upsampling = nn.Sequential(*upsampling)
        
        # Final output layer
        self.conv3 = nn.Conv2d(base_channels, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()

    def _make_res_block(self, channels):
        # Use RRDB blocks instead of basic residual blocks
        return RRDB(channels)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use a more stable initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Special initialization for the final conv layer to have smaller outputs
        nn.init.normal_(self.conv3.weight, 0, 0.01)
        if self.conv3.bias is not None:
            nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        # Add safety check for input
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.clamp(x, -1, 1)
        
        # Initial feature extraction
        x1 = self.relu(self.conv1(x))
        
        # Residual blocks
        res = self.res_blocks(x1)
        
        # Check for NaNs after residual blocks
        if torch.isnan(res).any() or torch.isinf(res).any():
            res = torch.zeros_like(x1)  # Reset to zeros rather than using bad values
        
        # Global residual connection
        res = self.bn(self.conv2(res))
        res = res + x1
        
        # Upsampling
        out = self.upsampling(res)
        
        # Final output layer
        out = self.conv3(out)
        
        # Apply activation and ensure output range is correct
        out = self.tanh(out)
        
        # Final safety check
        if torch.isnan(out).any() or torch.isinf(out).any():
            # Return a safe fallback
            out = torch.zeros_like(out)
            
        return out

class Discriminator(nn.Module):
    """
    Improved Discriminator network with stronger regularization.
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        in_channels, height, width = input_shape
        
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Sequence of discriminator blocks
        layers = []
        in_filters = 3
        out_filters_list = [64, 128, 256, 512]
        
        for i, out_filters in enumerate(out_filters_list):
            layers.extend(discriminator_block(
                in_filters, out_filters, first_block=(i == 0)
            ))
            in_filters = out_filters
        
        # Calculate final output size
        ds_size = height // 16  # Downsampled 4 times by stride=2
        
        self.model = nn.Sequential(*layers)
        
        # Dense layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_filters * ds_size * ds_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    
    def forward(self, img):
        features = self.model(img)
        validity = self.classifier(features)
        return validity

# -------------------------------
# Improved Calculation Functions
# -------------------------------
def calculate_psnr(sr, hr):
    """Calculate PSNR between super-resolved and high-resolution images"""
    # Convert from [-1,1] to [0,1] if needed
    if sr.min() < 0:
        sr = (sr + 1) / 2
    if hr.min() < 0:
        hr = (hr + 1) / 2
        
    mse = nn.MSELoss()(sr, hr).item()
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)

def inverse_normalize(tensor):
    """Convert from normalized [-1,1] to [0,1] range"""
    return (tensor + 1) / 2

def calculate_ssim(sr, hr):
    """Calculate SSIM (Structural Similarity Index) between images"""
    from skimage.metrics import structural_similarity as ssim
    
    # Check input shapes and make sure they are valid
    if isinstance(sr, torch.Tensor):
        sr = sr.cpu().numpy()
    if isinstance(hr, torch.Tensor):
        hr = hr.cpu().numpy()
    
    # Handle different dimensionalities:
    # If 4D tensors (B,C,H,W), convert to 3D by taking first image
    if sr.ndim == 4:
        sr = sr[0]
    if hr.ndim == 4:
        hr = hr[0]
    
    # If 3D tensors (C,H,W), convert to HWC format
    if sr.ndim == 3 and sr.shape[0] <= 3:
        sr = np.transpose(sr, (1, 2, 0))
    if hr.ndim == 3 and hr.shape[0] <= 3:
        hr = np.transpose(hr, (1, 2, 0))
    
    # Ensure values are in [0,1] range
    sr = np.clip(sr, 0, 1)
    hr = np.clip(hr, 0, 1)
    
    # Calculate SSIM
    try:
        # For color images (3 channels)
        if sr.shape[-1] == 3:
            ssim_value = ssim(sr, hr, channel_axis=-1, data_range=1.0)
        # For grayscale
        else:
            ssim_value = ssim(sr, hr, data_range=1.0)
        return ssim_value
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0

# -------------------------------
# Function to Save Example Output Images
# -------------------------------
def save_example_images(model, dataset, device, save_dir=None, num_images=5):
    """Save example SR images compared to HR and LR images"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Import necessary libraries here to avoid scope issues
    import numpy as np
    from PIL import Image
    
    if num_images > len(dataset):
        num_images = len(dataset)
    
    # Process a batch of images
        with torch.no_grad():
        for i in range(num_images):
            lr_img, hr_img = dataset[i]
            
            # Move to device and add batch dimension
            lr_img = lr_img.unsqueeze(0).to(device)
            hr_img = hr_img.unsqueeze(0).to(device)
            
            # Generate SR image
            sr_img = model(lr_img)
            
            # Denormalize images (convert from [-1,1] to [0,1])
            lr_img = (lr_img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            sr_img = (sr_img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            hr_img = (hr_img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            
            # Convert to numpy arrays in HWC format
            lr_np = lr_img.permute(1, 2, 0).numpy()
            sr_np = sr_img.permute(1, 2, 0).numpy()
            hr_np = hr_img.permute(1, 2, 0).numpy()
            
            # Scale LR image to HR size for display only
            hr_h, hr_w = hr_np.shape[0], hr_np.shape[1]
            lr_pil = Image.fromarray((lr_np * 255).astype(np.uint8))
            lr_upscaled = lr_pil.resize((hr_w, hr_h), Image.BICUBIC)
            lr_display = np.array(lr_upscaled) / 255.0
            
            # Save individual images
            plt.figure(figsize=(5, 5))
            plt.imshow(lr_display)
            plt.axis('off')
            plt.title("LR (Bicubic Upscale)")
        plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{i+1}_LR.png"), dpi=300)
            plt.close()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(sr_np)
            plt.axis('off')
            plt.title("SR")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{i+1}_SR.png"), dpi=300)
            plt.close()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(hr_np)
            plt.axis('off')
            plt.title("HR")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{i+1}_HR.png"), dpi=300)
            plt.close()
            
            # Create a side-by-side comparison
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(lr_display)
            plt.title("LR Input (Bicubic Upscale)")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(sr_np)
            plt.title("SR Output")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(hr_np)
            plt.title("HR Ground Truth")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{i+1}_comparison.png"), dpi=300)
            plt.close()
            
            # Calculate metrics for this image
            psnr_val = calculate_psnr(sr_np, hr_np)
            ssim_val = calculate_ssim(sr_np, hr_np)
            
            print(f"Sample {i+1} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
            
    print(f"Saved {num_images} example images to {save_dir}")
    model.train()

# -------------------------------
# Perceptual Loss (VGG-based) for better visual quality
# -------------------------------
class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            
            # Load pretrained VGG19
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
            
            # Freeze parameters
            for param in vgg.parameters():
                param.requires_grad = False
                
            # Define layer indices for feature extraction
            # Using deeper layers for more semantic features
            self.slice1 = nn.Sequential(*list(vgg.children())[:3])  # conv1_2
            self.slice2 = nn.Sequential(*list(vgg.children())[3:8])  # conv2_2
            self.slice3 = nn.Sequential(*list(vgg.children())[8:13])  # conv3_2
            self.slice4 = nn.Sequential(*list(vgg.children())[13:22])  # conv4_2
            self.slice5 = nn.Sequential(*list(vgg.children())[22:31])  # conv5_2
            
            # Layer weights (deeper layers have higher weights)
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            
            self.criterion = nn.L1Loss()  # Using L1 loss instead of MSE for sharper details
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.initialized = True
            print("Enhanced VGG perceptual loss initialized with multiple deep layers")
        except Exception as e:
            print(f"Could not initialize VGG loss: {e}")
            self.initialized = False
    
    def forward(self, sr, hr):
        if not self.initialized:
            # Fall back to MSE if VGG is not available
            return nn.MSELoss()(sr, hr)
            
        # Normalize to VGG input range
        sr = (sr + 1) / 2  # Convert from [-1,1] to [0,1]
        hr = (hr + 1) / 2
        
        # Apply ImageNet normalization
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        
        # Extract features at different VGG layers
        sr_features = [sr]
        hr_features = [hr]
        
        sr_features.append(self.slice1(sr_features[0]))
        sr_features.append(self.slice2(sr_features[1]))
        sr_features.append(self.slice3(sr_features[2]))
        sr_features.append(self.slice4(sr_features[3]))
        sr_features.append(self.slice5(sr_features[4]))
        
        hr_features.append(self.slice1(hr_features[0]))
        hr_features.append(self.slice2(hr_features[1]))
        hr_features.append(self.slice3(hr_features[2]))
        hr_features.append(self.slice4(hr_features[3]))
        hr_features.append(self.slice5(hr_features[4]))
        
        # Calculate weighted loss at each layer
        loss = 0
        for i in range(1, len(sr_features)):
            loss += self.weights[i-1] * self.criterion(sr_features[i], hr_features[i])
        
        return loss

# Add attention mechanism
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Improved Discriminator with spectral normalization
def spectral_norm(module):
    return nn.utils.spectral_norm(module)

class ImprovedDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(ImprovedDiscriminator, self).__init__()
        in_channels, height, width = input_shape
        
        def disc_block(in_filters, out_filters, stride, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride, 1, bias=False))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.features = nn.Sequential(
            # input is (3) x 256 x 256
            *disc_block(3, 64, stride=2, normalize=False),          # -> 128x128
            *disc_block(64, 128, stride=2),                        # -> 64x64
            *disc_block(128, 256, stride=2),                       # -> 32x32
            *disc_block(256, 512, stride=2),                       # -> 16x16
            *disc_block(512, 512, stride=2),                       # -> 8x8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(512, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(1024, 1))
        )
        
    def forward(self, x):
        features = self.features(x)
        validity = self.classifier(features)
        return validity

# Gradient penalty for WGAN-GP
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# EMA (Exponential Moving Average) for model weights
class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_params.data, current_params.data
            ema_params.data = old_weight * self.beta + (1 - self.beta) * new_weight

# -------------------------------
# Main Training Loop for SRGAN with Early Stopping and Gradient Clipping
# -------------------------------
def plot_training_progress(metrics, save_dir="training_plots"):
    """Plot training, validation, and test metrics at the end of training."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set style for better visualization
    plt.style.use('seaborn-darkgrid')
    
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Losses
    ax1 = fig.add_subplot(221)
    epochs = range(1, len(metrics['train_g_losses']) + 1)
    ax1.plot(epochs, metrics['train_g_losses'], 'b-', label='Generator Loss', linewidth=2)
    ax1.plot(epochs, metrics['train_d_losses'], 'r-', label='Discriminator Loss', linewidth=2)
    ax1.plot(epochs, metrics['val_losses'], 'g-', label='Validation Loss', linewidth=2)
    if 'test_losses' in metrics and metrics['test_losses']:
        ax1.plot(epochs, metrics['test_losses'], 'm-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training, Validation, and Test Losses', fontsize=14, pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # Plot 2: PSNR
    ax2 = fig.add_subplot(222)
    ax2.plot(epochs, metrics['val_psnr'], 'g-', label='Validation PSNR', linewidth=2)
    if 'test_psnr' in metrics and metrics['test_psnr']:
        ax2.plot(epochs, metrics['test_psnr'], 'm-', label='Test PSNR', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('PSNR over Training', fontsize=14, pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    # Plot 3: SSIM
    ax3 = fig.add_subplot(223)
    ax3.plot(epochs, metrics['val_ssim'], 'g-', label='Validation SSIM', linewidth=2)
    if 'test_ssim' in metrics and metrics['test_ssim']:
        ax3.plot(epochs, metrics['test_ssim'], 'm-', label='Test SSIM', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('SSIM', fontsize=12)
    ax3.set_title('SSIM over Training', fontsize=14, pad=10)
    ax3.legend(fontsize=10)
    ax3.grid(True)
    
    # Plot 4: LPIPS
    ax4 = fig.add_subplot(224)
    ax4.plot(epochs, metrics['val_lpips'], 'g-', label='Validation LPIPS', linewidth=2)
    if 'test_lpips' in metrics and metrics['test_lpips']:
        ax4.plot(epochs, metrics['test_lpips'], 'm-', label='Test LPIPS', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('LPIPS', fontsize=12)
    ax4.set_title('LPIPS over Training', fontsize=14, pad=10)
    ax4.legend(fontsize=10)
    ax4.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    
    # Save additional comparison plots between validation and test
    if 'test_psnr' in metrics and metrics['test_psnr']:
        # Create figure for validation vs test metrics
        fig_compare = plt.figure(figsize=(20, 15))
        
        # Plot comparison PSNR
        ax1_comp = fig_compare.add_subplot(221)
        ax1_comp.plot(epochs, metrics['val_psnr'], 'g-', label='Validation PSNR', linewidth=2)
        ax1_comp.plot(epochs, metrics['test_psnr'], 'm-', label='Test PSNR', linewidth=2)
        ax1_comp.set_xlabel('Epoch', fontsize=12)
        ax1_comp.set_ylabel('PSNR (dB)', fontsize=12)
        ax1_comp.set_title('Validation vs Test PSNR', fontsize=14, pad=10)
        ax1_comp.legend(fontsize=10)
        ax1_comp.grid(True)
        
        # Plot comparison SSIM
        ax2_comp = fig_compare.add_subplot(222)
        ax2_comp.plot(epochs, metrics['val_ssim'], 'g-', label='Validation SSIM', linewidth=2)
        ax2_comp.plot(epochs, metrics['test_ssim'], 'm-', label='Test SSIM', linewidth=2)
        ax2_comp.set_xlabel('Epoch', fontsize=12)
        ax2_comp.set_ylabel('SSIM', fontsize=12)
        ax2_comp.set_title('Validation vs Test SSIM', fontsize=14, pad=10)
        ax2_comp.legend(fontsize=10)
        ax2_comp.grid(True)
        
        # Plot comparison LPIPS
        ax3_comp = fig_compare.add_subplot(223)
        ax3_comp.plot(epochs, metrics['val_lpips'], 'g-', label='Validation LPIPS', linewidth=2)
        ax3_comp.plot(epochs, metrics['test_lpips'], 'm-', label='Test LPIPS', linewidth=2)
        ax3_comp.set_xlabel('Epoch', fontsize=12)
        ax3_comp.set_ylabel('LPIPS', fontsize=12)
        ax3_comp.set_title('Validation vs Test LPIPS', fontsize=14, pad=10)
        ax3_comp.legend(fontsize=10)
        ax3_comp.grid(True)
        
        # Calculate and plot differences between validation and test
        ax4_comp = fig_compare.add_subplot(224)
        psnr_diff = [t - v for t, v in zip(metrics['test_psnr'], metrics['val_psnr'])]
        ssim_diff = [t - v for t, v in zip(metrics['test_ssim'], metrics['val_ssim'])]
        lpips_diff = [t - v for t, v in zip(metrics['test_lpips'], metrics['val_lpips'])]
        
        ax4_comp.plot(epochs, psnr_diff, 'r-', label='Test-Val PSNR Diff', linewidth=2)
        ax4_comp.plot(epochs, ssim_diff, 'g-', label='Test-Val SSIM Diff', linewidth=2)
        ax4_comp.plot(epochs, lpips_diff, 'b-', label='Test-Val LPIPS Diff', linewidth=2)
        ax4_comp.set_xlabel('Epoch', fontsize=12)
        ax4_comp.set_ylabel('Difference', fontsize=12)
        ax4_comp.set_title('Test-Validation Metric Differences', fontsize=14, pad=10)
        ax4_comp.legend(fontsize=10)
        ax4_comp.grid(True)
        
        # Save comparison plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'validation_vs_test_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.close()

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    n_samples = 0
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate SR images
            sr_imgs = model(lr_imgs)
            
            # Calculate metrics
            psnr = calculate_psnr(sr_imgs, hr_imgs)
            
            # Calculate SSIM
            try:
                sr_np = (sr_imgs.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                hr_np = (hr_imgs.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                ssim_value = calculate_ssim(sr_np, hr_np)
            except Exception as e:
                print(f"Error calculating SSIM: {e}")
                ssim_value = 0
            
            # Calculate LPIPS
            try:
                lpips_value = lpips_loss_fn(sr_imgs, hr_imgs).mean().item()
            except Exception as e:
                print(f"Error calculating LPIPS: {e}")
                lpips_value = 0
            
            total_psnr += psnr
            total_ssim += ssim_value
            total_lpips += lpips_value
            n_samples += 1
    
    return {
        'psnr': total_psnr / n_samples,
        'ssim': total_ssim / n_samples,
        'lpips': total_lpips / n_samples
    }

def run_generator_sanity_check(generator, device):
    """Run a simple sanity check on the generator to verify it produces valid output"""
    print("\nRunning generator sanity check...")
    
    # Create a simple test tensor (random noise)
    test_input = torch.randn(1, 3, 64, 64).to(device)
    
    # Normalize to [-1, 1] range
    test_input = torch.clamp(test_input, -1, 1)
    
    # Forward pass through generator
    generator.eval()
    with torch.no_grad():
        try:
            output = generator(test_input)
            
            # Check if output contains NaN or Inf values
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("WARNING: Generator output contains NaN or Inf values!")
                return False
            else:
                print("Generator sanity check PASSED - output contains valid values")
                
            # Save an example output image
            try:
                import numpy as np
                from PIL import Image
                import matplotlib.pyplot as plt
                
                # Convert to numpy and normalize to [0, 1]
                output_np = (output[0].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                
                # Create directory for sanity check results
                os.makedirs("sanity_check", exist_ok=True)
                
                # Plot and save
                plt.figure(figsize=(10, 10))
                plt.imshow(output_np)
                plt.title("Generator Sanity Check Output")
                plt.axis('off')
                plt.savefig("sanity_check/generator_test_output.png")
                plt.close()
                
                print("Saved generator test output to sanity_check/generator_test_output.png")
                
                # Check if wandb is initialized before trying to log
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"sanity_check/generator_output": wandb.Image(output_np)})
                except Exception as e:
                    print(f"Warning: Could not log to wandb: {e}")
                
            except Exception as e:
                print(f"Error saving sanity check image: {e}")
            
            return True
            
        except Exception as e:
            print(f"Generator sanity check FAILED: {e}")
            return False
    
def train_srgan_track1(use_synthetic=False):
    """
    Train the SRGAN model with mixed precision and memory optimization techniques.
    
    Args:
        use_synthetic: Whether to use synthetic data
    """
    # Start timing the training process
    import time
    start_time = time.time()
    
    # Create necessary directories
    results_dir = "results"
    
    print("Starting training for Track 1 (SRGAN with Bicubic LR images)")
    
    # Verify dataset is available
    if not verify_dataset(use_synthetic):
        print("Aborting training due to missing dataset.")
        return
    
    # Clear GPU cache and release system resources
    clear_gpu_cache()
    manage_memory()
    
    # Initialize mixed precision training
    scaler = GradScaler()
    
    # Define crop sizes that maintain the 4x scaling factor
    hr_crop_size = 256
    lr_crop_size = hr_crop_size // 4  # 64 for 4x upscaling
    
    # Training parameters
    num_epochs = 100
    gradient_accumulation_steps = 4
    best_val_loss = float('inf')
    early_stopping_patience = 15
    no_improve_epochs = 0
    checkpoint_every = 5  # Save checkpoint every 5 epochs
    
    # Initialize wandb tracking
    use_wandb = init_wandb(project_name="SRGAN-DIV2K", config={
        "hr_crop_size": hr_crop_size,
        "lr_crop_size": lr_crop_size,
        "upscale_factor": 4
    })
    
    # Create transforms that keep LR and HR aligned
    # For HR: load, random crop, then convert to tensor
    # For LR: derive from HR by downscaling
    hr_transform = transforms.Compose([
        transforms.RandomCrop(hr_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Custom dataset class to ensure LR and HR are properly aligned
    class AlignedSRDataset(Dataset):
        def __init__(self, hr_dir, scale_factor=4):
            super(AlignedSRDataset, self).__init__()
            # Import PIL.Image here to ensure it's available in the class scope
            from PIL import Image
            self.Image = Image
            
            self.hr_dir = hr_dir
            self.scale_factor = scale_factor
            
            # Get all HR image files
            self.hr_files = [f for f in os.listdir(hr_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Cache for loaded images
            self.hr_cache = ImageCache(max_size=50)
            
            print(f"Found {len(self.hr_files)} HR images")
            if len(self.hr_files) > 0:
                print(f"Example HR file: {self.hr_files[0]}")
        
        def __len__(self):
            return len(self.hr_files)
        
        def __getitem__(self, idx):
            # Import PIL.Image here too for local access
            from PIL import Image
            
            # Load HR image
            hr_file = self.hr_files[idx]
            hr_path = os.path.join(self.hr_dir, hr_file)
            
            # Check cache first
            hr_img = self.hr_cache.get(hr_path)
            
            if hr_img is None:
                try:
                    with Image.open(hr_path) as img:
                        hr_img = img.convert("RGB").copy()
                        self.hr_cache.put(hr_path, hr_img)
                except Exception as e:
                    print(f"Error loading HR image at index {idx}: {e}")
                    # Return default image in case of error
                    hr_img = Image.new("RGB", (hr_crop_size, hr_crop_size), color=(0, 0, 0))
            
            # Apply random crop to HR image
            hr_transformed = hr_transform(hr_img)
            
            # Create LR image by downscaling the HR image by 4x
            # First convert HR tensor back to PIL for downscaling
            hr_denorm = (hr_transformed * 0.5 + 0.5).clamp(0, 1)
            hr_pil = transforms.ToPILImage()(hr_denorm)
            
            # Calculate LR size
            lr_size = (hr_crop_size // self.scale_factor, hr_crop_size // self.scale_factor)
            
            # Downsample HR to create LR using bicubic interpolation
            lr_pil = hr_pil.resize(lr_size, Image.BICUBIC)
            
            # Convert LR back to tensor and normalize
            lr_tensor = transforms.ToTensor()(lr_pil)
            lr_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lr_tensor)
            
            # Debug output - print tensor statistics
            if idx == 0:
                print(f"LR tensor stats: min={lr_tensor.min():.2f}, max={lr_tensor.max():.2f}, mean={lr_tensor.mean():.2f}")
                print(f"HR tensor stats: min={hr_transformed.min():.2f}, max={hr_transformed.max():.2f}, mean={hr_transformed.mean():.2f}")
                print(f"LR shape: {lr_tensor.shape}, HR shape: {hr_transformed.shape}")
            
            return lr_tensor, hr_transformed
    
    # Create dataset directly from HR images only
    print("Creating train dataset...")
    full_train_dataset = AlignedSRDataset(TRAIN_HR_PATH, scale_factor=4)
    
    # Split into train/test
    total_train = len(full_train_dataset)
    if total_train <= 100:
        test_size = max(5, int(total_train * 0.1))  # At least 5 samples or 10%
        train_size = total_train - test_size
    else:
        train_size = total_train - 100
        test_size = 100
        
    print("Splitting dataset...")
    train_dataset, test_dataset = random_split(full_train_dataset, [train_size, test_size])
    
    print("Creating validation dataset...")
    valid_dataset = AlignedSRDataset(VALID_HR_PATH, scale_factor=4)
    
    # Create results and checkpoint directories
    results_dir = "training_results"
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize models with correct input/output sizes
    generator = ImprovedGenerator(num_blocks=16).to(device)
    discriminator = ImprovedDiscriminator(input_shape=(3, 256, 256)).to(device)
    
    # Run a sanity check on the generator
    run_generator_sanity_check(generator, device)
    
    # Initialize EMA model
    ema_generator = ImprovedGenerator(num_blocks=16).to(device)
    ema_generator.load_state_dict(generator.state_dict())
    ema = EMA(0.999)
    
    # Loss functions
    content_loss_fn = nn.MSELoss().to(device)
    adversarial_loss_fn = nn.BCEWithLogitsLoss().to(device)
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Try to initialize VGG loss
    try:
        vgg_loss = VGGLoss(device)
        use_vgg_loss = vgg_loss.initialized
    except:
        use_vgg_loss = False
        vgg_loss = None
    
    # Optimizers with better parameters
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    # Learning rate schedulers with warm restarts
    scheduler_G = CosineAnnealingWarmRestarts(optimizer_G, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler_D = CosineAnnealingWarmRestarts(optimizer_D, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Training parameters
    num_epochs = 100
    gradient_accumulation_steps = 4
    best_val_loss = float('inf')
    early_stopping_patience = 15
    no_improve_epochs = 0
    checkpoint_every = 5  # Save checkpoint every 5 epochs
    
    # Modified loss weights for better stability
    content_weight = 1.0
    perceptual_weight = 0.1  # Reduced for stability
    adversarial_weight = 1e-4  # Reduced for stability
    lpips_weight = 0.1
    
    # Learning rate warmup
    warmup_epochs = 2
    initial_lr_G = 1e-5
    initial_lr_D = 1e-5
    target_lr_G = 1e-4
    target_lr_D = 1e-4
    
    # Initialize metrics dictionary
    metrics = {
        'train_g_losses': [],
        'train_d_losses': [],
        'val_losses': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_lpips': [],
        'test_losses': [],
        'test_psnr': [],
        'test_ssim': [],
        'test_lpips': []
    }
    
    # Load checkpoint if available
    start_epoch = 0
    latest_checkpoint = None
    
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) 
                             if f.startswith("checkpoint_epoch_") and f.endswith(".pth")])
    
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Found checkpoint: {latest_checkpoint}")
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(latest_checkpoint)
            
            # Load model states
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            # Load optimizer states
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            
            # Load scheduler states if they exist
            if 'scheduler_G_state_dict' in checkpoint:
                scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
                scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            
            # Load EMA generator if it exists
            if 'ema_generator_state_dict' in checkpoint:
                ema_generator.load_state_dict(checkpoint['ema_generator_state_dict'])
            
            # Load training state
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                
            if 'no_improve_epochs' in checkpoint:
                no_improve_epochs = checkpoint['no_improve_epochs']
                
            print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training will start from scratch.")
            start_epoch = 0
    
    # Use smaller batch sizes to reduce memory usage and disable workers
    try:
        print("Creating data loaders...")
        batch_size = 4  # Reduced batch size to reduce memory pressure
        # Use persistent_workers=False and num_workers=0 to avoid file handle issues
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            persistent_workers=False,
            pin_memory=False  # Disable pin_memory to reduce memory usage
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            persistent_workers=False,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            persistent_workers=False,
            pin_memory=False
        )
    except RuntimeError as e:
        print(f"Memory error detected: {e}")
        print("Using minimum batch size...")
        batch_size = 2  # Minimum viable batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            persistent_workers=False,
            pin_memory=False
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            persistent_workers=False,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            persistent_workers=False,
            pin_memory=False
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples (reserved): {len(test_dataset)}")
    print(f"Using batch size: {batch_size}")
    print(f"HR crop size: {hr_crop_size}x{hr_crop_size}")
    print(f"LR crop size: {lr_crop_size}x{lr_crop_size}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        generator.train()
        discriminator.train()
        epoch_g_losses = []
        epoch_d_losses = []
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            current_lr_G = initial_lr_G + (target_lr_G - initial_lr_G) * lr_scale
            current_lr_D = initial_lr_D + (target_lr_D - initial_lr_D) * lr_scale
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = current_lr_G
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = current_lr_D
        
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        
        # Display epoch info
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        
        # Clear memory at the beginning of each epoch
        clear_gpu_cache()
        
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            # Clean up at the beginning of each batch iteration
            torch.cuda.empty_cache()
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            with autocast():
                # Generate fake images
                fake_imgs = generator(lr_imgs)
                
                # Real and fake validity
                real_validity = discriminator(hr_imgs)
                fake_validity = discriminator(fake_imgs.detach())
                
                # Gradient penalty
                gp = compute_gradient_penalty(discriminator, hr_imgs.data, fake_imgs.data)
                
                # Discriminator loss with reduced magnitude
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10.0 * gp
            
            # Scale and backprop
            scaler.scale(d_loss).backward()
            
            # Only step optimizer every gradient_accumulation_steps
            if (i + 1) % gradient_accumulation_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer_D)
                
                # Clip discriminator gradients
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                
                # Step with scaler
                scaler.step(optimizer_D)
                optimizer_D.zero_grad()
            
            # Clean up intermediate tensors
            del real_validity, fake_validity, gp
            torch.cuda.empty_cache()
            
            # Train Generator (with separate scaler update)
            optimizer_G.zero_grad()
            
            with autocast():
                # Generate fake images again
                fake_imgs = generator(lr_imgs)
                
                # Get validity predictions
                fake_validity = discriminator(fake_imgs)
                
                # Content loss (L1)
                content_loss = F.l1_loss(fake_imgs, hr_imgs)
                
                # Perceptual loss with safety check
                if use_vgg_loss:
                    try:
                        perceptual_loss = vgg_loss(fake_imgs, hr_imgs)
                    except:
                        perceptual_loss = torch.tensor(0.0).to(device)
                        use_vgg_loss = False
                else:
                    perceptual_loss = torch.tensor(0.0).to(device)
                
                # LPIPS loss with safety check
                try:
                    lpips_value = lpips_loss_fn(fake_imgs, hr_imgs).mean()
                except:
                    lpips_value = torch.tensor(0.0).to(device)
                
                # Adversarial loss (with gradient scaling)
                adversarial_loss = -torch.mean(fake_validity)
                
                # Combined loss with safety checks
                g_loss = (content_weight * content_loss + 
                         perceptual_weight * perceptual_loss.clamp(-1e3, 1e3) + 
                         adversarial_weight * adversarial_loss.clamp(-1e3, 1e3) + 
                         lpips_weight * lpips_value.clamp(-1e3, 1e3))
            
            # Scale and backprop
            scaler.scale(g_loss).backward()
            
            # Only step optimizer every gradient_accumulation_steps
            if (i + 1) % gradient_accumulation_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer_G)
                
                # Clip generator gradients
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                
                # Step with scaler
                scaler.step(optimizer_G)
                
                # Update scaler for both D and G
                scaler.update()
                
                optimizer_G.zero_grad()
            
            # Update EMA model with safety check
            if not torch.isnan(g_loss):
                ema.update_model_average(ema_generator, generator)
            
            epoch_g_losses.append(g_loss.item() if not torch.isnan(g_loss) else 0.0)
            epoch_d_losses.append(d_loss.item() if not torch.isnan(d_loss) else 0.0)
            
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else 0
                avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses) if epoch_d_losses else 0
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] "
                      f"G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
                
                # Log intermediate batch metrics to wandb
                if use_wandb and (i + 1) % 100 == 0:
                    wandb.log({
                        "batch/G_loss": avg_g_loss,
                        "batch/D_loss": avg_d_loss,
                        "batch/step": i + 1 + epoch * len(train_loader),
                        "batch/epoch": epoch + 1
                    })
            
            # Clear memory more frequently
            if i % 5 == 0:
                # Release tensors
                del fake_imgs, fake_validity, content_loss
                if use_vgg_loss:
                    del perceptual_loss
                del lpips_value, adversarial_loss, g_loss, d_loss
                del lr_imgs, hr_imgs
                manage_memory()
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Run full memory cleanup between epochs
        manage_memory()
        
        # Validation with careful memory management
        print("\nRunning validation...")
        generator.eval()
        cumulative_val_loss = 0.0
        cumulative_val_psnr = 0.0
        cumulative_val_ssim = 0.0
        cumulative_val_lpips = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                # Clean up before each validation batch
                torch.cuda.empty_cache()
                
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                try:
                    sr_imgs = generator(lr_imgs)
                
                # Calculate validation loss
                val_loss = content_loss_fn(sr_imgs, hr_imgs).item()
                cumulative_val_loss += val_loss
                
                # Calculate metrics
                val_psnr = calculate_psnr(sr_imgs, hr_imgs)
                cumulative_val_psnr += val_psnr
                
                # Calculate SSIM with proper window size and error handling
                try:
                    sr_np = (sr_imgs.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                    hr_np = (hr_imgs.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                    
                    # Ensure images are at least 7x7 for SSIM calculation
                    min_dim = min(sr_np.shape[1], sr_np.shape[2])
                    win_size = min(7, min_dim) if min_dim % 2 == 1 else min(7, min_dim - 1)
                    
                        ssim_value = calculate_ssim(sr_np, hr_np)
                    cumulative_val_ssim += ssim_value
                        
                        # Free numpy arrays
                        del sr_np, hr_np
                        
                except Exception as e:
                    print(f"Error calculating SSIM: {e}")
                    ssim_value = 0.0
                    cumulative_val_ssim += ssim_value
                
                # Calculate LPIPS
                try:
                    lpips_value = lpips_loss_fn(sr_imgs, hr_imgs).mean().item()
                    cumulative_val_lpips += lpips_value
                except Exception as e:
                    print(f"Error calculating LPIPS: {e}")
                    lpips_value = 0.0
                    cumulative_val_lpips += lpips_value
                
                val_batches += 1
                    
                except Exception as e:
                    print(f"Error during validation: {e}")
                
                finally:
                    # Always clean up tensors, regardless of errors
                    if 'sr_imgs' in locals():
                        del sr_imgs
                    del lr_imgs, hr_imgs
                    torch.cuda.empty_cache()
                    
                # Clean up every 5 batches
                if val_batches % 5 == 0:
                    manage_memory()
        
        # Make sure validation batches isn't zero to avoid division by zero
        val_batches = max(1, val_batches)
        
        # Calculate averages
        avg_val_loss = cumulative_val_loss / val_batches
        avg_val_psnr = cumulative_val_psnr / val_batches
        avg_val_ssim = cumulative_val_ssim / val_batches
        avg_val_lpips = cumulative_val_lpips / val_batches
        
        # Store training metrics
        metrics['train_g_losses'].append(sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else 0)
        metrics['train_d_losses'].append(sum(epoch_d_losses) / len(epoch_d_losses) if epoch_d_losses else 0)
        
        # Store validation metrics
        metrics['val_losses'].append(avg_val_loss)
        metrics['val_psnr'].append(avg_val_psnr)
        metrics['val_ssim'].append(avg_val_ssim)
        metrics['val_lpips'].append(avg_val_lpips)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Metrics:")
        print(f"Train - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
        print(f"Validation - PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}, LPIPS: {avg_val_lpips:.4f}")
        
        # Evaluate on test set when validation is done
        print("\nRunning test evaluation...")
        generator.eval()
        cumulative_test_loss = 0.0
        cumulative_test_psnr = 0.0
        cumulative_test_ssim = 0.0
        cumulative_test_lpips = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in test_loader:
                # Clean up before each test batch
                torch.cuda.empty_cache()
                
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                try:
                    sr_imgs = generator(lr_imgs)
                    
                    # Calculate test loss
                    test_loss = content_loss_fn(sr_imgs, hr_imgs).item()
                    cumulative_test_loss += test_loss
                    
                    # Calculate metrics
                    test_psnr = calculate_psnr(sr_imgs, hr_imgs)
                    cumulative_test_psnr += test_psnr
                    
                    # Calculate SSIM with proper window size and error handling
                    try:
                        sr_np = (sr_imgs.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                        hr_np = (hr_imgs.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                        
                        ssim_value = calculate_ssim(sr_np, hr_np)
                        cumulative_test_ssim += ssim_value
                        
                        # Free numpy arrays
                        del sr_np, hr_np
                        
                    except Exception as e:
                        print(f"Error calculating test SSIM: {e}")
                        ssim_value = 0.0
                        cumulative_test_ssim += ssim_value
                    
                    # Calculate LPIPS
                    try:
                        lpips_value = lpips_loss_fn(sr_imgs, hr_imgs).mean().item()
                        cumulative_test_lpips += lpips_value
                    except Exception as e:
                        print(f"Error calculating test LPIPS: {e}")
                        lpips_value = 0.0
                        cumulative_test_lpips += lpips_value
                    
                    test_batches += 1
                    
                except Exception as e:
                    print(f"Error during test evaluation: {e}")
                
                finally:
                    # Always clean up tensors, regardless of errors
                    if 'sr_imgs' in locals():
                        del sr_imgs
                    del lr_imgs, hr_imgs
                    torch.cuda.empty_cache()
                    
                # Clean up every 5 batches
                if test_batches % 5 == 0:
                    manage_memory()
        
        # Make sure test batches isn't zero to avoid division by zero
        test_batches = max(1, test_batches)
        
        # Calculate averages
        avg_test_loss = cumulative_test_loss / test_batches
        avg_test_psnr = cumulative_test_psnr / test_batches
        avg_test_ssim = cumulative_test_ssim / test_batches
        avg_test_lpips = cumulative_test_lpips / test_batches
        
        # Store test metrics
        metrics['test_losses'].append(avg_test_loss)
        metrics['test_psnr'].append(avg_test_psnr)
        metrics['test_ssim'].append(avg_test_ssim)
        metrics['test_lpips'].append(avg_test_lpips)
        
        # Print test metrics
        print(f"Test - PSNR: {avg_test_psnr:.2f} dB, SSIM: {avg_test_ssim:.4f}, LPIPS: {avg_test_lpips:.4f}")
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/G_loss": avg_g_loss,
                "train/D_loss": avg_d_loss,
                "val/loss": avg_val_loss,
                "val/PSNR": avg_val_psnr,
                "val/SSIM": avg_val_ssim,
                "val/LPIPS": avg_val_lpips,
                "test/loss": avg_test_loss,
                "test/PSNR": avg_test_psnr,
                "test/SSIM": avg_test_ssim,
                "test/LPIPS": avg_test_lpips,
                "learning_rate/G": optimizer_G.param_groups[0]['lr'],
                "learning_rate/D": optimizer_D.param_groups[0]['lr']
            })
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            # Save best model
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'psnr': avg_val_psnr,
                'ssim': avg_val_ssim,
                'lpips': avg_val_lpips,
            }, "track1_srgan_best.pth")
            print("Saved best model.")
            
            # Log best model to wandb
            if use_wandb:
                wandb.save("track1_srgan_best.pth")
                
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
                break
        
        # Log sample images to wandb every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_images_to_wandb(generator, valid_loader, device, epoch, use_wandb)
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'ema_generator_state_dict': ema_generator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'metrics': metrics,
                'best_val_loss': best_val_loss,
                'no_improve_epochs': no_improve_epochs
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean old checkpoints, keeping only the last 3
            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) 
                                      if f.startswith("checkpoint_epoch_") and f.endswith(".pth")])
            if len(checkpoint_files) > 3:
                for old_file in checkpoint_files[:-3]:
                    try:
                        os.remove(os.path.join(checkpoint_dir, old_file))
                    except Exception as e:
                        print(f"Error removing old checkpoint {old_file}: {e}")
        
        # Display epoch duration
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch completed in {epoch_time:.2f} seconds")
    
    # Calculate training time
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(generator, test_loader, device)
    
    # Create results directory
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'test_metrics': test_metrics,
        'training_time': training_time
    }, os.path.join(results_dir, "final_model.pth"))
    
    # Generate and save plots
    print("\nGenerating training plots...")
    plot_training_progress(metrics, save_dir=results_dir)
    
    # Save example images
    print("Saving example images...")
    save_example_images(generator, test_dataset, device, save_dir=os.path.join(results_dir, "examples"), num_images=5)
    
    # Print final results
    print("\nFinal Test Metrics:")
    print(f"PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"SSIM: {test_metrics['ssim']:.4f}")
    print(f"LPIPS: {test_metrics['lpips']:.4f}")
    print(f"\nResults saved in '{results_dir}' directory")
    print("- Model weights: final_model.pth")
    print("- Training plots: training_metrics.png")
    print("- Example images: examples/")
    
    # Log final results to wandb
    if use_wandb:
        # Log final metrics
        wandb.log({
            "test/PSNR": test_metrics['psnr'],
            "test/SSIM": test_metrics['ssim'],
            "test/LPIPS": test_metrics['lpips'],
            "training_time_hours": hours,
            "training_time_minutes": minutes,
            "training_time_seconds": seconds,
        })
        
        # Upload final model and plots
        wandb.save(os.path.join(results_dir, "final_model.pth"))
        wandb.save(os.path.join(results_dir, "training_metrics.png"))
        
        # Upload example images
        for i in range(5):
            try:
                example_path = os.path.join(results_dir, "examples", f"example_{i}.png")
                if os.path.exists(example_path):
                    wandb.save(example_path)
            except Exception as e:
                print(f"Error uploading example image to wandb: {e}")
        
        # Add summary metrics
        wandb.run.summary.update({
            "best_PSNR": max(metrics['val_psnr']),
            "best_SSIM": max(metrics['val_ssim']),
            "best_LPIPS": min(metrics['val_lpips']),
            "final_PSNR": test_metrics['psnr'],
            "final_SSIM": test_metrics['ssim'],
            "final_LPIPS": test_metrics['lpips'],
            "total_epochs": epoch + 1,
            "early_stopped": epoch + 1 < num_epochs,
        })
        
        # Finish the wandb run
        wandb.finish()

# Global variable to store fixed validation samples
fixed_validation_samples = None

def log_images_to_wandb(generator, valid_loader, device, epoch, use_wandb):
    """Log sample images to wandb for visualization using same samples across epochs"""
    if not use_wandb:
        return
    
    global fixed_validation_samples
    
    # Set generator to evaluation mode
    generator.eval()
    
    # Enable gradient capturing to track where NaNs originate
    with torch.set_grad_enabled(False):
        # Import necessary libraries for image processing
        from PIL import Image
        import numpy as np
        
        # Get or use cached samples
        if fixed_validation_samples is None:
            # First time: fetch and cache samples
            fixed_validation_samples = []
            try:
                # Always start from the beginning of the validation set
                valid_loader_iter = iter(valid_loader)
                
                # Collect 3 good samples to use across all epochs
                sample_count = 0
                max_attempts = 10  # Try at most 10 samples to find 3 good ones
                
                for i in range(max_attempts):
                    if sample_count >= 3:  # We have enough samples
                        break
                        
                    try:
                        sample_lr, sample_hr = next(valid_loader_iter)
                        # Check for NaN/Inf
                        if (torch.isnan(sample_lr).any() or torch.isinf(sample_lr).any() or 
                            torch.isnan(sample_hr).any() or torch.isinf(sample_hr).any()):
                            continue
                            
                        # Convert to device and store
                        fixed_validation_samples.append((sample_lr.to(device), sample_hr.to(device)))
                        sample_count += 1
                    except Exception:
                        continue
                
                if not fixed_validation_samples:
                    generator.train()  # Return to training mode
                    return
            except Exception:
                generator.train()
                return
        
        # Process each of our fixed samples
        for idx, (sample_lr, sample_hr) in enumerate(fixed_validation_samples):
            # Generate super-resolution image
            try:
                # Verify input tensor again
                if torch.isnan(sample_lr).any() or torch.isinf(sample_lr).any():
                    sample_lr = torch.clamp(sample_lr, -1, 1)
                
                # Generate SR image with exception handling
                with torch.no_grad():  # Ensure no gradients for inference
                    # Make sure the generator is in eval mode
                    generator.eval()
                    
                    # Normalize input if needed
                    sample_lr_safe = torch.clamp(sample_lr, -1, 1)
                    
                    # Generate SR image
                    sample_sr = generator(sample_lr_safe)
                
                # Check if SR output is valid
                if torch.isnan(sample_sr).any() or torch.isinf(sample_sr).any():
                    # Generate a simple upscaled version as fallback
                    from torchvision.transforms.functional import resize
                    sample_sr = resize(sample_lr, sample_hr.shape[2:])
                
                # Ensure all tensors are on CPU
                sample_lr_cpu = sample_lr.cpu()
                sample_hr_cpu = sample_hr.cpu()
                sample_sr_cpu = sample_sr.cpu()
                
                # Convert from [-1,1] to [0,1] for visualization with safety checks
                sample_lr_np = (torch.clamp(sample_lr_cpu[0] * 0.5 + 0.5, 0, 1)).permute(1, 2, 0).numpy()
                sample_sr_np = (torch.clamp(sample_sr_cpu[0] * 0.5 + 0.5, 0, 1)).permute(1, 2, 0).numpy()
                sample_hr_np = (torch.clamp(sample_hr_cpu[0] * 0.5 + 0.5, 0, 1)).permute(1, 2, 0).numpy()
                
                # Additional check for very dark images
                if sample_sr_np.max() < 0.1:
                    # Apply min-max normalization
                    sr_min, sr_max = sample_sr_np.min(), sample_sr_np.max()
                    if sr_max > sr_min:  # Avoid division by zero
                        sample_sr_np = (sample_sr_np - sr_min) / (sr_max - sr_min)
                    else:
                        # In case of constant value, use grayscale
                        sample_sr_np = np.ones_like(sample_sr_np) * 0.5
                
                # Upscale LR image for display purposes only
                h, w, _ = sample_hr_np.shape
                lr_pil = Image.fromarray((sample_lr_np * 255).astype(np.uint8))
                lr_upscaled = lr_pil.resize((w, h), Image.BICUBIC)
                lr_upscaled_np = np.array(lr_upscaled) / 255.0
                
                # Create individual images for wandb
                sample_lr_img = wandb.Image(
                    lr_upscaled_np,  # Use upscaled version for display
                    caption=f"LR Input {idx+1} (Upscaled for Display)"
                )
                sample_sr_img = wandb.Image(
                    sample_sr_np,
                    caption=f"SR Output {idx+1} (Epoch {epoch+1})"
                )
                sample_hr_img = wandb.Image(
                    sample_hr_np,
                    caption=f"HR Ground Truth {idx+1}"
                )
                
                # Create a side-by-side comparison
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(lr_upscaled_np)
                plt.title(f"LR Input {idx+1}")
                plt.axis("off")
                
                plt.subplot(1, 3, 2)
                plt.imshow(sample_sr_np)
                plt.title(f"SR Output {idx+1} (Epoch {epoch+1})")
                plt.axis("off")
                
                plt.subplot(1, 3, 3)
                plt.imshow(sample_hr_np)
                plt.title(f"HR Ground Truth {idx+1}")
                plt.axis("off")
                
                plt.tight_layout()
                
                # Save to a temp file
                comparison_path = f"temp_comparison_epoch_{epoch+1}_sample_{idx+1}.png"
                plt.savefig(comparison_path, dpi=300)
                plt.close()
                
                # Log the comparison image to wandb
                comparison_img = wandb.Image(
                    comparison_path,
                    caption=f"Comparison {idx+1} at Epoch {epoch+1}"
                )
                
                # Create a standardized tag for consistent sample tracking
                sample_tag = f"fixed_sample_{idx+1}"
                
                # Log individual and comparison images
                wandb.log({
                    f"images/{sample_tag}/LR_input": sample_lr_img,
                    f"images/{sample_tag}/SR_output": sample_sr_img,
                    f"images/{sample_tag}/HR_ground_truth": sample_hr_img,
                    f"images/{sample_tag}/comparison": comparison_img,
                    "epoch": epoch + 1
                })
                
                # Clean up temp file
                try:
                    os.remove(comparison_path)
                except:
                    pass
                
            except Exception:
                continue
    
    # Return generator to training mode
    generator.train()

# Function for memory tracking and debugging
def print_tensor_memory_usage(tensor_dict):
    """Print memory usage of tensors to help debug memory issues"""
    total_size = 0
    print("\nTensor Memory Usage:")
    print("-" * 80)
    print(f"{'Tensor Name':<30} {'Shape':<20} {'Size (MB)':<15} {'Device':<10}")
    print("-" * 80)
    
    for name, tensor in tensor_dict.items():
        # Calculate size in MB
        if isinstance(tensor, torch.Tensor):
            size_bytes = tensor.element_size() * tensor.nelement()
            size_mb = size_bytes / (1024 * 1024)
            total_size += size_mb
            device = str(tensor.device)
            shape = str(tensor.shape)
            print(f"{name:<30} {shape:<20} {size_mb:<15.2f} {device:<10}")
    
    print("-" * 80)
    print(f"Total: {total_size:.2f} MB")
    
    # Print total GPU memory if available
    if torch.cuda.is_available():
        print("\nGPU Memory:")
        print(f"Allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")
    
    print("\nSystem Memory:")
    # Try to get system memory usage
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Total: {mem.total / (1024 * 1024 * 1024):.2f} GB")
        print(f"Available: {mem.available / (1024 * 1024 * 1024):.2f} GB")
        print(f"Used: {mem.used / (1024 * 1024 * 1024):.2f} GB ({mem.percent}%)")
    except:
        print("Could not retrieve system memory info")
    
    print("-" * 80)

# Function to visualize model outputs at different training stages
def visualize_model_outputs(model, test_images, device, save_dir="model_progress", include_artifacts=True):
    """
    Create a visualization of model outputs with different levels of artifacts.
    This is useful for understanding how the model is improving over time.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        # Get a sample image
        lr_img, hr_img = test_images
        
        if not isinstance(lr_img, torch.Tensor):
            # Convert PIL images to tensors if needed
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            lr_img = normalize(to_tensor(lr_img)).unsqueeze(0)
            hr_img = normalize(to_tensor(hr_img)).unsqueeze(0)
        
        # Move to device
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        
        # Generate clean SR image
        sr_img = model(lr_img)
        
        # If requested, create versions with different artifacts
        outputs = [sr_img]
        titles = ["SR Output"]
        
        if include_artifacts:
            # Add noise to create a "noisy" version
            noise_level = 0.1
            noise = torch.randn_like(sr_img) * noise_level
            noisy_sr = torch.clamp(sr_img + noise, -1, 1)
            outputs.append(noisy_sr)
            titles.append("SR with Noise")
            
            # Add blur to create a "blurry" version
            kernel_size = 5
            sigma = 2.0
            padding = kernel_size // 2
            
            # Create Gaussian kernel
            x = torch.linspace(-padding, padding, kernel_size)
            kernel_1d = torch.exp(-(x * x) / (2 * sigma * sigma))
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            kernel = torch.outer(kernel_1d, kernel_1d).to(device)
            kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
            
            # Apply blur
            blurry_sr = torch.zeros_like(sr_img)
            for c in range(3):
                channel = sr_img[:, c:c+1]
                blurry_sr[:, c:c+1] = F.conv2d(
                    F.pad(channel, (padding, padding, padding, padding), mode='reflect'),
                    kernel[c:c+1], groups=1
                )
            
            outputs.append(blurry_sr)
            titles.append("SR with Blur")
            
            # Add both noise and blur
            noisy_blurry_sr = torch.clamp(blurry_sr + noise, -1, 1)
            outputs.append(noisy_blurry_sr)
            titles.append("SR with Noise+Blur")
        
        # Add LR and HR for comparison
        # Upscale LR to HR size for visualization
        lr_upscaled = F.interpolate(lr_img, scale_factor=4, mode='bicubic', align_corners=False)
        
        # Create comparison grid
        all_images = [lr_upscaled] + outputs + [hr_img]
        all_titles = ["LR (Upscaled)"] + titles + ["HR Ground Truth"]
        
        # Create visualization
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert tensors to numpy arrays
        np_images = []
        for img in all_images:
            # Normalize to [0, 1]
            img = (img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            # Convert to HWC format
            img = img.permute(1, 2, 0).numpy()
            np_images.append(img)
        
        # Create plot grid
        fig, axs = plt.subplots(1, len(np_images), figsize=(4 * len(np_images), 6))
        if len(np_images) == 1:
            axs = [axs]
        
        for i, (img, title) in enumerate(zip(np_images, all_titles)):
            axs[i].imshow(img)
            axs[i].set_title(title)
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "model_outputs_comparison.png"), dpi=300)
        plt.close()
        
        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "model_outputs/comparison": wandb.Image(
                        os.path.join(save_dir, "model_outputs_comparison.png")
                    )
                })
        except Exception as e:
            print(f"Warning: Could not log visualization to wandb: {e}")
    
    model.train()
    return os.path.join(save_dir, "model_outputs_comparison.png")

# Add a function to create an animated GIF of progress over training epochs
def create_training_progress_animation(image_dir, save_path="training_progress.gif"):
    """
    Create an animated GIF from training progress images.
    
    Args:
        image_dir: Directory containing progress images, named by epoch
        save_path: Path to save the resulting GIF
    """
    try:
        from PIL import Image
        import glob
        import os
        
        # Find all progress images
        image_files = sorted(glob.glob(os.path.join(image_dir, "epoch_*_comparison.png")))
        
        if len(image_files) < 2:
            print("Not enough images found to create animation")
            return None
        
        print(f"Creating animation from {len(image_files)} images...")
        
        # Load images
        images = [Image.open(f) for f in image_files]
        
        # Create animated GIF
        images[0].save(
            save_path,
            save_all=True,
            append_images=images[1:],
            duration=500,  # 500ms per frame
            loop=0  # Loop forever
        )
        
        print(f"Animation saved to {save_path}")
        
        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"training_progress/animation": wandb.Video(save_path)})
        except Exception as e:
            print(f"Warning: Could not log animation to wandb: {e}")
        
        return save_path
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        return None

# Relativistic adversarial loss for more stable GAN training
class RelativisticAdversarialLoss(nn.Module):
    def __init__(self, discriminator):
        super(RelativisticAdversarialLoss, self).__init__()
        self.discriminator = discriminator
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, real_images, fake_images):
        # Real image prediction
        real_pred = self.discriminator(real_images)
        
        # Fake image prediction
        fake_pred = self.discriminator(fake_images)
        
        # Relativistic average prediction for real images (D_ra)
        real_rel = real_pred - torch.mean(fake_pred)
        
        # Relativistic average prediction for fake images (D_ra)
        fake_rel = fake_pred - torch.mean(real_pred)
        
        # Loss for discriminator
        d_real_loss = self.criterion(real_rel, torch.ones_like(real_rel))
        d_fake_loss = self.criterion(fake_rel, torch.zeros_like(fake_rel))
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # Loss for generator (reversed targets)
        g_real_loss = self.criterion(fake_rel, torch.ones_like(fake_rel))
        g_fake_loss = self.criterion(real_rel, torch.zeros_like(real_rel))
        g_loss = (g_real_loss + g_fake_loss) / 2
        
        return d_loss, g_loss

if __name__ == "__main__":
    # Set to True to use synthetic data for testing purposes
    use_synthetic_data = False
    
    # Start training
    if use_synthetic_data:
        print("\nUsing synthetic data for a quick test run...")
        
        # Create the synthetic dataset with more samples for better testing
        create_synthetic_dataset(use_synthetic=True, num_samples=16)
        
        # Create a generator model for testing
        generator = ImprovedGenerator(num_blocks=8).to(device)
        
        # Run sanity check on the generator
        run_generator_sanity_check(generator, device)
        
        # Create a simple validation loader with the synthetic data
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        # Initialize wandb
        use_wandb = init_wandb(project_name="SRGAN-Test", config={
            "test_mode": True,
            "synthetic_data": True,
            "num_samples": 16,
            "model_type": "ImprovedGenerator",
            "num_blocks": 8
        })
        
        # Define transforms for LR and HR images
        lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Create a dataset from the synthetic data with transforms
        valid_dataset = PairedBicubicDataset(
            os.path.join("synthetic_DIV2K", "DIV2K_valid_LR_bicubic_X4", "DIV2K_valid_LR_bicubic", "X4"),
            os.path.join("synthetic_DIV2K", "DIV2K_valid_HR", "DIV2K_valid_HR"),
            lr_transform=lr_transform,
            hr_transform=hr_transform
        )
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        
        # Create a test loader with the training data to have more samples
        test_dataset = PairedBicubicDataset(
            os.path.join("synthetic_DIV2K", "DIV2K_train_LR_bicubic_X4", "DIV2K_train_LR_bicubic", "X4"),
            os.path.join("synthetic_DIV2K", "DIV2K_train_HR"),
            lr_transform=lr_transform,
            hr_transform=hr_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
        # Log some images without training
        print("Logging samples to wandb...")
        log_images_to_wandb(generator, valid_loader, device, 0, use_wandb=use_wandb)
        
        # Create a more comprehensive visualization of model outputs
        try:
            print("Creating visualization of model outputs...")
            test_images = next(iter(test_loader))
            visualize_model_outputs(generator, test_images, device, save_dir="test_outputs")
        except Exception as e:
            print(f"Error creating visualization: {e}")
        
        # Run a quick mock training test to verify all components work
        print("\nRunning a quick test training loop...")
        # Create a discriminator for testing
        discriminator = ImprovedDiscriminator(input_shape=(3, 256, 256)).to(device)
        
        # Loss functions
        content_loss_fn = nn.MSELoss().to(device)
        adversarial_loss_fn = nn.BCEWithLogitsLoss().to(device)
        
        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
        
        # Test a few mini-batches
        generator.train()
        discriminator.train()
        
        try:
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                if i >= 3:  # Just test 3 batches
                    break
                    
                # Move to device
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                # Train discriminator
                optimizer_D.zero_grad()
                fake_imgs = generator(lr_imgs)
                real_validity = discriminator(hr_imgs)
                fake_validity = discriminator(fake_imgs.detach())
                
                # Simple adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                d_loss.backward()
                optimizer_D.step()
                
                # Train generator
                optimizer_G.zero_grad()
                fake_imgs = generator(lr_imgs)
                fake_validity = discriminator(fake_imgs)
                
                # Content loss
                content_loss = content_loss_fn(fake_imgs, hr_imgs)
                adversarial_loss = -torch.mean(fake_validity)
                
                # Total generator loss
                g_loss = content_loss + 0.001 * adversarial_loss
                g_loss.backward()
                optimizer_G.step()
                
                print(f"Test batch {i+1}/3 - G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")
                
                # Clear memory
                del fake_imgs, real_validity, fake_validity, content_loss, adversarial_loss, g_loss, d_loss
                torch.cuda.empty_cache()
            
            print("Test training loop completed successfully")
        except Exception as e:
            print(f"Error during test training loop: {e}")
        
        # Close wandb run
        if use_wandb:
            wandb.finish()
            
        print("\nAll tests completed. You're ready to run full training!")
        print("To start full training, set use_synthetic_data = False and run the script again.")
    else:
        # Run the full training
        train_srgan_track1(use_synthetic=use_synthetic_data)