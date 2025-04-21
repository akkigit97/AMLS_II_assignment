import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast  # Updated import path
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torchvision.transforms as transforms
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
import time
import wandb
import os.path
import platform
import math
import sys
import argparse
import traceback
import json
from PIL import Image, ImageFilter
from torchvision import models
from torchvision.models import vgg19
from tqdm import tqdm
from contextlib import nullcontext

# Import torch.cuda.amp module as amp
import torch.cuda.amp as amp

# For Google Colab compatibility
IS_COLAB = False
try:
    import google.colab
    IS_COLAB = True
    print("Google Colab detected")
    
    # The following will only execute when running in Colab
    # Setup Colab environment
    def setup_colab():
        try:
            print("Setting up Colab environment...")
            # Mount Google Drive
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Install any missing packages
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "lpips", "wandb"])
            
            # Set the working directory for Colab
            import os
            os.chdir('/content')
            
            # Set paths for Google Colab
            global COLAB_BASE_DIR
            COLAB_BASE_DIR = '/content'
            print(f"Working directory set to: {os.getcwd()}")
        except Exception as e:
            print(f"Error setting up Colab environment: {e}")
    
    # We don't run setup immediately to avoid side effects during import
    # It will be called in the main function if needed
            
except ImportError:
    IS_COLAB = False

# Try to import resource module (Linux/Mac only)
try:
    import resource
except ImportError:
    # Create a dummy resource module for Windows
    class DummyResource:
        RLIMIT_NOFILE = 0
        
        @staticmethod
        def getrlimit(limit):
            return (1024, 4096)
            
        @staticmethod
        def setrlimit(limit, limits):
            pass
    
    resource = DummyResource()

# -------------------------------------
# Device configuration
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# For CPU training, reduce model size
CPU_MODE = device.type == 'cpu'
COLAB_GPU_MODE = IS_COLAB and device.type == 'cuda'

if CPU_MODE:
    print("CPU mode detected: Using reduced model size and optimized settings")
    DEFAULT_NUM_BLOCKS = 4  # Reduced from 16
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_EPOCHS = 20     # Reduced for testing
    PROGRESS_FREQUENCY = 1  # Show progress more frequently
elif COLAB_GPU_MODE:
    print("Colab GPU mode detected: Using optimized settings for Colab")
    DEFAULT_NUM_BLOCKS = 16  # Full model size for GPU
    DEFAULT_BATCH_SIZE = 8   # Colab GPUs typically have enough memory for larger batches
    DEFAULT_EPOCHS = 50      # Reduced slightly since Colab sessions are limited
    PROGRESS_FREQUENCY = 2   # Show progress more frequently than default but less than CPU
else:
    DEFAULT_NUM_BLOCKS = 16
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_EPOCHS = 100
    PROGRESS_FREQUENCY = 5

# Initialize wandb tracking
def init_wandb(project_name="SRGAN-Track2", config=None):
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

# Try to import psutil for memory monitoring
try:
    import psutil
except ImportError:
    try:
        os.system("pip install psutil")
        import psutil
    except:
        # Create dummy psutil for systems where it can't be installed
        class DummyProcess:
            def memory_info(self):
                class MemInfo:
                    def __init__(self):
                        self.rss = 0
                        self.vms = 0
                return MemInfo()
            
            def open_files(self):
                return []
                
        class DummyPsutil:
            def Process(self, _):
                return DummyProcess()
        
        psutil = DummyPsutil()
        print("Warning: psutil not available. Memory monitoring will be limited.")

# Increase file limit for Linux/Mac systems
def increase_file_limit():
    """Increase the maximum number of open files on Linux/Mac"""
    if platform.system() != 'Windows':
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            new_soft = min(hard, 4096)  # Increase to hard limit or 4096, whichever is smaller
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"File limit increased from {soft} to {new_soft}")
        except Exception as e:
            print(f"Could not increase file limit: {e}")

# Print resource usage for debugging
def print_resource_usage():
    """Print current memory and file usage"""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        print("\nResource Usage:")
        print(f"RAM Usage: {mem_info.rss / 1024 / 1024:.2f} MB")
        
        if hasattr(process, 'open_files'):
            open_files = process.open_files()
            print(f"Open files: {len(open_files)}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB allocated, "
                  f"{torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB reserved")
    except Exception as e:
        print(f"Error monitoring resources: {e}")

# Function to clear GPU cache
def clear_gpu_cache():
    """Clear GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU cache cleared")

# Comprehensive memory management
def manage_memory():
    """Run full memory cleanup routine"""
    # Run garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # On Windows, we can try a more aggressive approach
    if platform.system() == 'Windows':
        import ctypes
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    
    # Print current memory usage if in debug mode
    # print_resource_usage()

# Simple image cache to reduce file I/O
class ImageCache:
    """Cache for loaded images to reduce file I/O"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.keys = []
    
    def get(self, path):
        """Get image from cache if it exists"""
        if path in self.cache:
            # Move to end of keys list (most recently used)
            self.keys.remove(path)
            self.keys.append(path)
            return self.cache[path]
        return None
    
    def put(self, path, img):
        """Add image to cache, remove least recently used if full"""
        if path not in self.cache:
            # Handle cache size limit
            if len(self.keys) >= self.max_size:
                # Remove least recently used
                old_key = self.keys.pop(0)
                del self.cache[old_key]
            
            # Add new item
            self.cache[path] = img
            self.keys.append(path)

# -------------------------------------
# Folder Paths for Unknown Degradation (Track 2)
# -------------------------------------
if IS_COLAB:
    # Colab paths
    TRAIN_LR_UNKNOWN_PATH = "/content/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown/X4"
    TRAIN_HR_PATH         = "/content/DIV2K/DIV2K_train_HR"
    VALID_LR_UNKNOWN_PATH = "/content/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown/X4"
    VALID_HR_PATH         = "/content/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR"
else:
    # Use relative paths that work on both Windows and Linux
    TRAIN_LR_UNKNOWN_PATH = os.path.join("DIV2K", "DIV2K_train_LR_unknown_X4", "DIV2K_train_LR_unknown", "X4")
    TRAIN_HR_PATH         = os.path.join("DIV2K", "DIV2K_train_HR")
    VALID_LR_UNKNOWN_PATH = os.path.join("DIV2K", "DIV2K_valid_LR_unknown_X4", "DIV2K_valid_LR_unknown", "X4")
    VALID_HR_PATH         = os.path.join("DIV2K", "DIV2K_valid_HR", "DIV2K_valid_HR")

# Function to detect DIV2K directory structure
def verify_and_update_dataset_paths(lr_path=None, hr_path=None, verbose=True):
    """
    Check if dataset paths exist and try to detect correct paths if not found.
    Args:
        lr_path: Optional manual override for LR path
        hr_path: Optional manual override for HR path
        verbose: Whether to print detailed information
    Returns:
        Dictionary with verified paths or None if paths are invalid
    """
    global TRAIN_LR_UNKNOWN_PATH, TRAIN_HR_PATH, VALID_LR_UNKNOWN_PATH, VALID_HR_PATH
    
    # Check if manual paths were provided and override defaults
    if lr_path:
        TRAIN_LR_UNKNOWN_PATH = lr_path
        if verbose:
            print(f"Using manual LR path: {TRAIN_LR_UNKNOWN_PATH}")
    
    if hr_path:
        TRAIN_HR_PATH = hr_path
        if verbose:
            print(f"Using manual HR path: {TRAIN_HR_PATH}")
    
    # Try to find dataset directories
    if not os.path.exists(TRAIN_LR_UNKNOWN_PATH):
        if verbose:
            print(f"Warning: LR path not found at {TRAIN_LR_UNKNOWN_PATH}")
            print("Attempting to detect correct path...")
        
        # Try common alternative structures and the exact paths provided by the user
        alt_paths = [
            # Exact paths from the user
            "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown/X4",
            
            # Common alternatives
            os.path.join(os.getcwd(), "DIV2K_train_LR_unknown_X4"),
            os.path.join(os.getcwd(), "DIV2K", "DIV2K_train_LR_unknown_X4"),
            os.path.join(os.getcwd(), "DIV2K", "DIV2K_train_LR_unknown_X4", "DIV2K_train_LR_unknown", "X4"),
            
            # Look for parent directories
            os.path.dirname(TRAIN_LR_UNKNOWN_PATH),
            os.path.join(os.getcwd(), "..", "DIV2K", "DIV2K_train_LR_unknown_X4"),
        ]
        
        for path in alt_paths:
            if os.path.exists(path) and os.path.isdir(path):
                TRAIN_LR_UNKNOWN_PATH = path
                # Update valid path based on train path pattern
                if "train" in path:
                    VALID_LR_UNKNOWN_PATH = path.replace("train", "valid")
                if verbose:
                    print(f"Found alternative LR path: {TRAIN_LR_UNKNOWN_PATH}")
                break
    
    if not os.path.exists(TRAIN_HR_PATH):
        if verbose:
            print(f"Warning: HR path not found at {TRAIN_HR_PATH}")
            print("Attempting to detect correct path...")
        
        # Try common alternative structures and the exact paths provided by the user
        alt_paths = [
            # Exact paths from the user
            "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_train_HR",
            
            # Common alternatives
            os.path.join(os.getcwd(), "DIV2K_train_HR"),
            os.path.join(os.getcwd(), "DIV2K", "DIV2K_train_HR"),
            
            # Look for parent directories
            os.path.dirname(TRAIN_HR_PATH),
            os.path.join(os.getcwd(), "..", "DIV2K", "DIV2K_train_HR"),
        ]
        
        for path in alt_paths:
            if os.path.exists(path) and os.path.isdir(path):
                TRAIN_HR_PATH = path
                # Update valid path based on train path pattern
                if "train" in path:
                    valid_path = path.replace("train", "valid")
                    # Check if valid path exists and has a nested structure like in the provided paths
                    nested_valid_path = os.path.join(valid_path, os.path.basename(valid_path))
                    if os.path.exists(nested_valid_path) and os.path.isdir(nested_valid_path):
                        VALID_HR_PATH = nested_valid_path
                    else:
                        VALID_HR_PATH = valid_path
                if verbose:
                    print(f"Found alternative HR path: {TRAIN_HR_PATH}")
                break
    
    # Check the specific validation paths separately
    if not os.path.exists(VALID_LR_UNKNOWN_PATH):
        specific_valid_path = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown/X4"
        if os.path.exists(specific_valid_path) and os.path.isdir(specific_valid_path):
            VALID_LR_UNKNOWN_PATH = specific_valid_path
            if verbose:
                print(f"Using specific validation LR path: {VALID_LR_UNKNOWN_PATH}")
    
    if not os.path.exists(VALID_HR_PATH):
        specific_valid_path = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR"
        if os.path.exists(specific_valid_path) and os.path.isdir(specific_valid_path):
            VALID_HR_PATH = specific_valid_path
            if verbose:
                print(f"Using specific validation HR path: {VALID_HR_PATH}")
    
    # Final verification
    paths_valid = os.path.exists(TRAIN_LR_UNKNOWN_PATH) and os.path.exists(TRAIN_HR_PATH)
    
    if paths_valid:
        # Count files to verify dataset looks valid
        lr_files = [f for f in os.listdir(TRAIN_LR_UNKNOWN_PATH) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        hr_files = [f for f in os.listdir(TRAIN_HR_PATH)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(lr_files) > 0 and len(hr_files) > 0:
            if verbose:
                print(f"Dataset verified - Found {len(lr_files)} LR images and {len(hr_files)} HR images")
            return {
                'TRAIN_LR_UNKNOWN_PATH': TRAIN_LR_UNKNOWN_PATH,
                'TRAIN_HR_PATH': TRAIN_HR_PATH,
                'VALID_LR_UNKNOWN_PATH': VALID_LR_UNKNOWN_PATH,
                'VALID_HR_PATH': VALID_HR_PATH
            }
        else:
            if verbose:
                print(f"Paths exist but found {len(lr_files)} LR images and {len(hr_files)} HR images")
            return None
    else:
        if verbose:
            print("Failed to find valid dataset paths")
            print(f"LR path exists: {os.path.exists(TRAIN_LR_UNKNOWN_PATH)}")
            print(f"HR path exists: {os.path.exists(TRAIN_HR_PATH)}")
        return None

# -------------------------------------
# Improved Dataset Class for Paired Unknown Data (Track 2)
# -------------------------------------
class PairedUnknownDataset(Dataset):
    """Dataset for unknown degradation super-resolution with improved pairing logic"""
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None, patch_size=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.lr_cache = ImageCache(max_size=50)  # Cache for LR images
        self.hr_cache = ImageCache(max_size=50)  # Cache for HR images
        
        # Debug paths
        print(f"LR directory: {lr_dir}")
        print(f"HR directory: {hr_dir}")
        
        # Get all image files
        lr_files = sorted([f for f in os.listdir(lr_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        hr_files = sorted([f for f in os.listdir(hr_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(lr_files)} LR files and {len(hr_files)} HR files")
        
        # Print a few examples to check naming patterns
        if lr_files:
            print(f"LR file examples: {lr_files[:3]}")
        if hr_files:
            print(f"HR file examples: {hr_files[:3]}")
        
        # Extract IDs for matching - handle different naming conventions
        self.paired_files = []
        
        # Build a lookup from ID to filename for both LR and HR
        lr_ids = {}
        hr_ids = {}
        
        # Parse LR filenames - handle format like "0001x4.png"
        for lr_file in lr_files:
            # Extract numeric ID, handling different patterns
            clean_id = self._extract_clean_id(lr_file)
            if clean_id:
                lr_ids[clean_id] = lr_file
        
        # Parse HR filenames - typically like "0001.png"
        for hr_file in hr_files:
            # Extract numeric ID, handling different patterns
            clean_id = self._extract_clean_id(hr_file)
            if clean_id:
                hr_ids[clean_id] = hr_file
        
        # Find common IDs with debugging
        common_ids = set(lr_ids.keys()) & set(hr_ids.keys())
        if common_ids:
            print(f"Found {len(common_ids)} matching image pairs out of {len(lr_files)} LR and {len(hr_files)} HR images")
        else:
            print("ERROR: No matching image pairs found!")
            print("LR IDs:", sorted(list(lr_ids.keys()))[:5])
            print("HR IDs:", sorted(list(hr_ids.keys()))[:5])
            
            # Emergency fallback pairing if no matches found
            if len(lr_files) > 0 and len(hr_files) > 0:
                print("Using emergency fallback pairing based on sorted order")
                max_pairs = min(len(lr_files), len(hr_files))
                for i in range(max_pairs):
                    self.paired_files.append((lr_files[i], hr_files[i]))
        
        # If we found common IDs, create proper pairs
        if common_ids:
            for img_id in sorted(common_ids):
                self.paired_files.append((lr_ids[img_id], hr_ids[img_id]))
                
            # Print paired file examples
            if self.paired_files:
                print("Paired examples:")
                for i, (lr, hr) in enumerate(self.paired_files[:3]):
                    print(f"  Pair {i+1}: LR={lr}, HR={hr}")
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
    
    def _extract_clean_id(self, filename):
        """Extract a clean numerical ID from filename, handling different patterns"""
        # First, try to extract any sequence of digits
        match = re.search(r'(\d+)', filename)
        if not match:
            return None
            
        # Get the numeric part
        numeric_id = match.group(1)
        
        # Remove any scale indicator (like x4) if present
        clean_id = re.sub(r'x\d+$', '', numeric_id)
        
        # Convert to integer to ensure proper matching
        try:
            return int(clean_id)
        except ValueError:
            return None
    
    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        """Get a paired LR and HR item with robust error handling"""
        try:
            # Get filenames for this pair
            lr_file, hr_file = self.paired_files[idx]
            lr_path = os.path.join(self.lr_dir, lr_file)
            hr_path = os.path.join(self.hr_dir, hr_file)
            
            # Check if files exist
            if not os.path.exists(lr_path):
                print(f"LR file not found: {lr_path}")
                raise FileNotFoundError(f"LR file not found: {lr_path}")
                
            if not os.path.exists(hr_path):
                print(f"HR file not found: {hr_path}")
                raise FileNotFoundError(f"HR file not found: {hr_path}")
            
            # First check cache
            lr_img = self.lr_cache.get(lr_path)
            hr_img = self.hr_cache.get(hr_path)
            
            # If not in cache, load from disk
            if lr_img is None or hr_img is None:
                # Load images with proper file handling
                try:
                    with Image.open(lr_path) as img:
                        lr_img = img.convert("RGB").copy()
                        # Store original size
                        lr_size = lr_img.size
                        self.lr_cache.put(lr_path, lr_img)
                except Exception as e:
                    print(f"Error loading LR image {lr_path}: {e}")
                    # Create a default LR image
                    lr_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
                
                try:
                    with Image.open(hr_path) as img:
                        hr_img = img.convert("RGB").copy()
                        # Store original size
                        hr_size = hr_img.size
                        self.hr_cache.put(hr_path, hr_img)
                except Exception as e:
                    print(f"Error loading HR image {hr_path}: {e}")
                    # Create a default HR image
                    hr_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
                
                # Verify scale factor (should be 4x)
                hr_width, hr_height = hr_size
                lr_width, lr_height = lr_size
                
                # Ensure HR is exactly 4x the size of LR (or very close to it)
                expected_hr_width = lr_width * 4
                expected_hr_height = lr_height * 4
                
                # Debug sizes
                if abs(hr_width - expected_hr_width) > 4 or abs(hr_height - expected_hr_height) > 4:
                    print(f"Size mismatch for pair {idx} - LR: {lr_size}, HR: {hr_size}, Expected HR: ({expected_hr_width}, {expected_hr_height})")
                    
                    # Fix if sizes don't match the expected 4x factor
                    hr_img = hr_img.resize((lr_width * 4, lr_height * 4), Image.BICUBIC)
                    print(f"  → Resized HR to: {hr_img.size}")
            
            # Apply transforms - with safety checks
            if self.hr_transform:
                try:
                    hr_tensor = self.hr_transform(hr_img)
                except Exception as e:
                    print(f"Error applying HR transform: {e}")
                    # Create a fallback tensor
                    hr_tensor = torch.zeros((3, 256, 256))
            else:
                # Convert to tensor if no transform is provided
                hr_tensor = torch.tensor(np.array(hr_img).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            
            if self.lr_transform:
                try:
                    lr_tensor = self.lr_transform(lr_img)
                except Exception as e:
                    print(f"Error applying LR transform: {e}")
                    # Create a fallback tensor
                    lr_tensor = torch.zeros((3, 64, 64))
            else:
                # Convert to tensor if no transform is provided
                lr_tensor = torch.tensor(np.array(lr_img).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            
            # Ensure tensor shapes are correct (perform a sanity check)
            if isinstance(lr_tensor, torch.Tensor) and isinstance(hr_tensor, torch.Tensor):
                # Check channel count
                if lr_tensor.shape[0] != 3 or hr_tensor.shape[0] != 3:
                    print(f"Channel count mismatch - LR: {lr_tensor.shape}, HR: {hr_tensor.shape}")
                
                # LR should be 1/4 the size of HR in each dimension
                c_lr, h_lr, w_lr = lr_tensor.shape
                c_hr, h_hr, w_hr = hr_tensor.shape
                
                # Verify that HR is approximately 4x the size of LR
                if abs(h_hr - h_lr * 4) > 2 or abs(w_hr - w_lr * 4) > 2:
                    print(f"Size mismatch after transform - LR: {lr_tensor.shape}, HR: {hr_tensor.shape}")
                    
                    # Since transforms are applied, we can't easily resize here,
                    # but we can log the issue for debugging
            
            # Final quality check - replace NaN or Inf values
            if torch.isnan(lr_tensor).any() or torch.isinf(lr_tensor).any():
                print(f"Warning: NaN or Inf values in LR tensor at index {idx}")
                lr_tensor = torch.nan_to_num(lr_tensor, nan=0.0, posinf=1.0, neginf=0.0)
            
            if torch.isnan(hr_tensor).any() or torch.isinf(hr_tensor).any():
                print(f"Warning: NaN or Inf values in HR tensor at index {idx}")
                hr_tensor = torch.nan_to_num(hr_tensor, nan=0.0, posinf=1.0, neginf=0.0)
            
            return lr_tensor, hr_tensor
            
        except Exception as e:
            print(f"Unhandled error when loading sample at index {idx}: {e}")
            # Return a dummy pair as fallback
            return torch.zeros((3, 64, 64)), torch.zeros((3, 256, 256))

# -------------------------------------
# Improved Model Components
# -------------------------------------
class AdaptiveNorm(nn.Module):
    """Adaptive normalization layer that can handle varying spatial dimensions"""
    def __init__(self, num_features):
        super(AdaptiveNorm, self).__init__()
        self.norm = nn.GroupNorm(8, num_features)  # Group normalization is invariant to input size
    
    def forward(self, x):
        return self.norm(x)

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

class ImprovedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ImprovedResidualBlock, self).__init__()
        
        # Simplified block with batch normalization for stability
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)  # Use ReLU instead of PReLU for stability
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Initialize with small values
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Explicitly avoid in-place addition to prevent numerical issues
        return out

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
    """Residual in Residual Dense Block"""
    def __init__(self, num_features, growth_channels=32, num_convs=5, res_scale=0.2):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(num_features, growth_channels, num_convs)
        self.RDB2 = RDB(num_features, growth_channels, num_convs)
        self.RDB3 = RDB(num_features, growth_channels, num_convs) 
        self.beta = nn.Parameter(torch.tensor(res_scale))  # Learnable residual scaling parameter
        
        # Add adaptive smoothing capability
        self.smooth_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.smooth_factor = nn.Parameter(torch.tensor(0.15))  # Control smoothing intensity
        
        # Initialize with smaller weights to reduce artifacts
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                m.weight.data *= 0.1  # Scale down weights to reduce artifacts
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        
        # Apply adaptive smoothing to reduce high-frequency artifacts
        smooth_out = self.smooth_conv(out)
        smooth_out = F.avg_pool2d(smooth_out, kernel_size=3, stride=1, padding=1)
        out = (1 - self.smooth_factor) * out + self.smooth_factor * smooth_out
        
        # Apply learnable beta factor to control residual intensity
        # Use a sigmoid activation to keep beta in a reasonable range (0-1)
        effective_beta = torch.sigmoid(self.beta) * 0.5  # Limit max to 0.5 to prevent oversharpening
        
        return x + effective_beta * out

class RDB(nn.Module):
    """Residual Dense Block for RRDB"""
    def __init__(self, num_features, growth_channels=32, num_convs=5, scaling_factor=0.2):
        super(RDB, self).__init__()
        self.num_features = num_features
        self.num_convs = num_convs
        self.growth_channels = growth_channels
        self.scaling_factor = scaling_factor
        
        # Define convolution layers with growth connections
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(num_features, growth_channels, kernel_size=3, padding=1))
        
        for i in range(1, num_convs):
            in_channels = num_features + i * growth_channels
            self.convs.append(nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1))
            
        self.local_beta = nn.Parameter(torch.tensor(scaling_factor))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.final_conv = nn.Conv2d(num_features + num_convs * growth_channels, 
                                   num_features, kernel_size=1)
    
    def forward(self, x):
        inputs = [x]
        for i in range(self.num_convs):
            out = self.convs[i](torch.cat(inputs, 1))
            out = self.lrelu(out)
            inputs.append(out)
            
        out = self.final_conv(torch.cat(inputs, 1))
        return x + out * self.local_beta

class ImprovedGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_channels=64, num_blocks=16, upscale_factor=4):
        super(ImprovedGenerator, self).__init__()
        
        # Use a more stable architecture with fewer blocks for initial training
        num_blocks = min(num_blocks, 8)  # Limit block count to reduce complexity initially
        
        # Define model layers
        # Initial convolution block with lower channel count
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1)
        
        # Residual blocks with activation scaling
        self.residual_blocks = nn.ModuleList([
            ImprovedResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # Skip connection with batch norm for stability
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        
        # Upsampling blocks with batch norm
        if upscale_factor == 4:
            self.upsampling = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels * 4),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels * 4),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        else:
            # Simplified for now - add more options as needed
            raise ValueError(f"Upscale factor {upscale_factor} not supported")
            
        # Final output layer with small initialization
        self.conv_output = nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Careful initialization
        self._initialize_weights()
        
        # Scaling factor for stability
        self.residual_scale = 0.1
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use smaller standard deviation for initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize the final layer with smaller weights for stability
        nn.init.normal_(self.conv_output.weight, std=0.001)
        nn.init.constant_(self.conv_output.bias, 0)
        
    def forward(self, x):
        # Add check for NaN inputs
        if torch.isnan(x).any():
            print("Warning: NaN detected in generator input")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Store input for residual connection
        out1 = self.conv_input(x)
        
        # Process through residual blocks with scaling factor
        out = out1.clone()
        for block in self.residual_blocks:
            out = out + self.residual_scale * block(out)  # Scale residual to prevent overflow
        
        # Skip connection
        out = self.conv_mid(out)
        out = out + out1
        
        # Upsampling
        out = self.upsampling(out)
        
        # Final convolution with clipping
        out = self.conv_output(out)
        
        # Use tanh with safety clipping
        return torch.tanh(torch.clamp(out, -10, 10))

def spectral_norm(module):
    return nn.utils.spectral_norm(module)

class ImprovedDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(ImprovedDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True, stride=2):
            """Returns layers of each discriminator block"""
            layers = []
            layers.append(spectral_norm(nn.Conv2d(
                in_filters, 
                out_filters, 
                kernel_size=3, 
                stride=stride, 
                padding=1
            )))
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.input_shape = input_shape
        channels, height, width = input_shape
        
        # More gradual downsampling with more blocks and smaller stride
        self.models = nn.Sequential(
            # First block - no normalization
            *discriminator_block(channels, 64, normalization=False, stride=1),
            # Use spectral norm and batch norm with moderate feature growth
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # Final projection
            spectral_norm(nn.Linear(512, 1))
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        # Add mild noise during training for stability
        if self.training:
            noise = torch.randn_like(img) * 0.05
            img = img + noise
        
        return self.models(img)

# Gradient penalty for WGAN-GP
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculate gradient penalty for WGAN-GP"""
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

# Perceptual Loss (VGG-based)
class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        
        # Load pretrained VGG model - use a simpler version (VGG16 instead of VGG19)
        vgg = models.vgg16(pretrained=True).to(device)
        vgg.eval()
        
        # Use fewer layers to reduce complexity and chance of instability
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Use L1 loss - more stable than MSE for perceptual loss
        self.criterion = nn.L1Loss()
        self.device = device
        
    def forward(self, sr, hr):
        # Safe normalization with clipping to prevent extreme values
        sr = torch.clamp(sr, -1.0, 1.0)
        hr = torch.clamp(hr, -1.0, 1.0)
        
        # Normalize inputs from [-1, 1] to [0, 1]
        sr = (sr + 1.0) / 2.0
        hr = (hr + 1.0) / 2.0
        
        # Extract features
        sr_features = self.features(sr)
        hr_features = self.features(hr)
        
        # Compute loss
        loss = self.criterion(sr_features, hr_features)
        
        # Check for NaN values and return zero if found (to prevent training collapse)
        if torch.isnan(loss):
            print("Warning: NaN detected in VGG loss, returning zero loss")
            return torch.zeros(1, device=self.device, requires_grad=True)
            
        return loss

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
    
    # Create a fixed set of indices to use consistently across epochs
    # Store these indices in a global variable to ensure the same samples are used
    global fixed_example_indices
    if fixed_example_indices is None or len(fixed_example_indices) == 0:
        # Initialize with evenly spaced indices
        dataset_size = len(dataset)
        fixed_example_indices = [i * dataset_size // num_images for i in range(num_images)]
        # Make sure we don't exceed dataset size
        fixed_example_indices = [min(i, dataset_size-1) for i in fixed_example_indices]
        print(f"Fixed example indices: {fixed_example_indices}")
    
    # Process a batch of images
    with torch.no_grad():
        for i, idx in enumerate(fixed_example_indices[:num_images]):
            # Get paired LR and HR images
            try:
                lr_img, hr_img = dataset[idx]
                
                # Debug info
                print(f"Sample {i+1} (index {idx}) shapes - LR: {lr_img.shape}, HR: {hr_img.shape}")
                
                # Move to device and add batch dimension
                lr_img = lr_img.unsqueeze(0).to(device)
                hr_img = hr_img.unsqueeze(0).to(device)
                
                # Generate SR image
                sr_img = model(lr_img)
                
                # Check for NaN/Inf
                if torch.isnan(sr_img).any() or torch.isinf(sr_img).any():
                    print(f"Warning: SR output contains NaN/Inf for sample {idx}")
                    from torchvision.transforms.functional import resize
                    sr_img = resize(lr_img, size=[hr_img.shape[2], hr_img.shape[3]])
                
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
                plt.title("LR (Unknown Degradation)")
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
                plt.title("LR Input (Unknown Degradation)")
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
                psnr_val = calculate_psnr(sr_img, hr_img)
                ssim_val = calculate_ssim(sr_np, hr_np)
                
                print(f"Sample {i+1} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
            except Exception as e:
                print(f"Error processing example sample {i+1} (index {idx}): {e}")
            
    print(f"Saved {num_images} example images to {save_dir}")
    model.train()

def save_compact_comparison(model, dataset, device, save_dir="compact_comparisons", num_images=5):
    """
    Save compact side-by-side comparison images (LR vs SR vs HR) without borders or titles
    to match the reference style shown in the example.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Use fewer samples if dataset is smaller
    num_images = min(num_images, len(dataset))
    
    # Process images
    with torch.no_grad():
        for i in range(num_images):
            try:
                # Get paired images
                lr_img, hr_img = dataset[i]
                
                # Debug info
                print(f"Processing comparison image {i+1}")
                
                # Move to device and add batch dimension
                lr_img = lr_img.unsqueeze(0).to(device)
                hr_img = hr_img.unsqueeze(0).to(device)
                
                # Generate SR image
                sr_img = model(lr_img)
                
                # Calculate metrics on tensor versions before converting to numpy
                try:
                    psnr_val = calculate_psnr(sr_img, hr_img)
                    # Keep tensor versions for SSIM calculation
                    sr_tensor = sr_img.clone().squeeze(0).cpu()
                    hr_tensor = hr_img.clone().squeeze(0).cpu()
                    # Convert to numpy for SSIM calculation
                    sr_np_for_metrics = (sr_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
                    hr_np_for_metrics = (hr_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
                    ssim_val = calculate_ssim(sr_np_for_metrics, hr_np_for_metrics)
                    print(f"Sample {i+1} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
                except Exception as e:
                    print(f"Error calculating metrics: {e}")
                
                # Denormalize images (convert from [-1,1] to [0,1])
                lr_img = (lr_img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
                sr_img = (sr_img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
                hr_img = (hr_img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
                
                # Convert to numpy arrays in HWC format
                lr_np = lr_img.permute(1, 2, 0).numpy()
                sr_np = sr_img.permute(1, 2, 0).numpy()
                hr_np = hr_img.permute(1, 2, 0).numpy()
                
                # Debug info
                print(f"Image shapes - LR: {lr_np.shape}, SR: {sr_np.shape}, HR: {hr_np.shape}")
                
                # Scale LR image to HR size for display
                hr_h, hr_w = hr_np.shape[0], hr_np.shape[1]
                lr_pil = Image.fromarray((lr_np * 255).astype(np.uint8))
                lr_upscaled = lr_pil.resize((hr_w, hr_h), Image.BICUBIC)
                lr_display = np.array(lr_upscaled) / 255.0
                
                # Create clean, compact comparison (no titles, no axes)
                # Create a figure with 3 subplots arranged horizontally with minimal spacing
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                plt.subplots_adjust(wspace=0.01, hspace=0)  # Minimal spacing between images
                
                # Plot each image without axes or borders
                axes[0].imshow(lr_display)
                axes[0].axis('off')
                
                axes[1].imshow(sr_np)
                axes[1].axis('off')
                
                axes[2].imshow(hr_np)
                axes[2].axis('off')
                
                # Tight layout with minimal white space
                plt.tight_layout(pad=0.1)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                # Save the compact comparison
                plt.savefig(os.path.join(save_dir, f"compact_comparison_{i+1}.png"), 
                           bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Error processing comparison image {i+1}: {e}")
    
    print(f"Saved {num_images} compact comparison images to {save_dir}")
    model.train()

# Function to create directories if they don't exist
def create_dir(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def train_unknown_srgan(use_synthetic=False, use_wandb=False, verbose=True,
                       batch_size=16, num_epochs=200, save_every=1,
                       lr_gen=1e-4, lr_disc=1e-4,
                       patch_size=24, device='cuda'):
    """
    Train SRGAN with mixed precision and memory optimizations.
    """
    
    global TRAIN_LR_UNKNOWN_PATH, TRAIN_HR_PATH, VALID_LR_UNKNOWN_PATH, VALID_HR_PATH
    
    # Create directories if they don't exist
    create_dir("checkpoints")
    create_dir("results_track2")
    
    # Set device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        # Free up GPU memory
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        if device == 'cuda':
            print("CUDA not available, using CPU instead")
    
    # Initialize mixed precision training - use the standard API without custom wrappers
    use_amp = device.type == 'cuda'  # Only use mixed precision on CUDA
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb_run = init_wandb()
    
    # Set different patch sizes based on device
    if device.type == 'cpu':
        patch_size = 16  # Smaller patch size for CPU
    
    # Apply transformations
    train_hr_transforms = transforms.Compose([
        transforms.CenterCrop(size=(patch_size * 4, patch_size * 4)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_lr_transforms = transforms.Compose([
        transforms.CenterCrop(size=(patch_size, patch_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create validation transforms - no augmentation for validation
    valid_hr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    valid_lr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PairedUnknownDataset(
        TRAIN_LR_UNKNOWN_PATH, TRAIN_HR_PATH,
        lr_transform=train_lr_transforms,
        hr_transform=train_hr_transforms,
        patch_size=patch_size
    )
    
    valid_dataset = PairedUnknownDataset(
        VALID_LR_UNKNOWN_PATH, VALID_HR_PATH,
        lr_transform=valid_lr_transform,
        hr_transform=valid_hr_transform
    )
    
    # Create dataloaders
    print(f"Creating dataloaders with batch size {batch_size}...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1,  # Use batch_size=1 to handle variable-sized images
        shuffle=False, num_workers=0, pin_memory=False
    )
    
    # ... existing code ...
    
    # Initialize models
    generator = ImprovedGenerator(in_channels=3, out_channels=3, num_channels=64, num_blocks=16, upscale_factor=4).to(device)
    discriminator = ImprovedDiscriminator(input_shape=(3, 96, 96)).to(device)
    
    # Initialize models with proper initialization
    generator = ImprovedGenerator(in_channels=3, out_channels=3, num_channels=64, num_blocks=16, upscale_factor=4).to(device)
    # Apply improved weight initialization to generator
    for m in generator.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    discriminator = ImprovedDiscriminator(input_shape=(3, 96, 96)).to(device)
    
    # Load VGG for perceptual loss with frozen parameters
    vgg = vgg19(pretrained=True).features[:36].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    
    # ... existing code ...
    
    # Initialize optimizers with improved parameters
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.9, 0.999))
    
    # Learning rate schedulers - use cosine annealing for smoother decay
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=1e-6)
    
    # Initialize optimizers with lower learning rates for stability
    lr_gen_stable = 5e-5  # Lower learning rate
    lr_disc_stable = 5e-5  # Lower learning rate
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_gen_stable, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_disc_stable, betas=(0.9, 0.999))
    
    # Initialize optimizers with extremely low learning rates to fix NaN issues
    lr_gen_stable = 1e-6  # Drastically reduced learning rate
    lr_disc_stable = 1e-6  # Drastically reduced learning rate
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_gen_stable, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_disc_stable, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    
    # Use a more gentle learning rate schedule
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)
    
    # Loss functions
    content_criterion = nn.L1Loss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    vgg_loss = VGGLoss(device=device).to(device)
    
    # LPIPS loss for perceptual quality
    try:
        import lpips
        lpips_loss = lpips.LPIPS(net='alex').to(device)
        use_lpips = True
        print("Using LPIPS loss")
    except:
        use_lpips = False
        print("LPIPS not available, skipping")
    
    # Early stopping parameters
    early_stopping_patience = 15
    no_improve_epochs = 0
    best_val_psnr = 0.0
    
    # Create metrics tracking dictionary
    metrics = {
        'train_g_losses': [],
        'train_d_losses': [],
        'val_losses': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_lpips': [],
    }
    
    # Training loop
    if verbose:
        print("Starting training...")
    
    best_psnr = 0.0
    best_ssim = 0.0
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        # Initialize loss tracking
        epoch_gen_adv_loss = 0.0
        epoch_gen_content_loss = 0.0
        epoch_gen_perceptual_loss = 0.0
        epoch_gen_lpips_loss = 0.0
        epoch_disc_loss = 0.0
        batch_count = 0
        
        # Progressive loss weights - optimized for clearer images with fewer artifacts
        progress_ratio = min(epoch / (num_epochs * 0.5), 1.0)  # Faster transition to perceptual losses
        
        # Content loss weight - maintain high weight for pixel accuracy
        content_weight = 1.0
        
        # Perceptual weight - gradually increase to enhance details
        # Start lower and increase more significantly later in training
        perceptual_weight = 0.1 + 0.5 * progress_ratio
        
        # Adversarial weight - very careful increase to avoid artifacts
        # Start extremely low and increase slowly
        adversarial_weight = 0.0001 + progress_ratio * 0.001
        
        # LPIPS weight - for additional perceptual quality if available
        lpips_weight = 0.05 * progress_ratio if use_lpips else 0.0
        
        # Use fixed, conservative loss weights for the first few epochs to stabilize training
        # After the initial stable period, gradually adjust weights
        stable_period = 5  # First 5 epochs use safe weights
        
        if epoch < stable_period:
            # Very conservative weights during initial stabilization
            content_weight = 1.0      # Focus mainly on content loss for stability
            perceptual_weight = 0.05  # Very small perceptual loss
            adversarial_weight = 0.0  # No adversarial component
            lpips_weight = 0.0        # No LPIPS to start
        else:
            # After stabilization, carefully increase weights
            progress_ratio = min((epoch - stable_period) / (num_epochs * 0.5), 1.0)
            content_weight = 1.0
            perceptual_weight = 0.05 + 0.2 * progress_ratio  # Slower increase
            adversarial_weight = 0.00005 * progress_ratio    # Very cautious
            lpips_weight = 0.01 * progress_ratio if use_lpips else 0.0
        
        # Make training extremely stable at the beginning - focus only on L1 reconstruction
        # Then slowly add other loss components
        warmup_period = 10  # First 10 epochs for pure reconstruction
        
        if epoch < warmup_period:
            # First 10 epochs: pure L1 reconstruction loss only
            content_weight = 1.0
            perceptual_weight = 0.0  # No perceptual loss
            adversarial_weight = 0.0  # No adversarial component
            lpips_weight = 0.0      # No LPIPS
        elif epoch < warmup_period + 10:
            # Next 10 epochs: add tiny bit of perceptual loss
            content_weight = 1.0
            perceptual_weight = 0.01  # Very minimal perceptual loss
            adversarial_weight = 0.0   # Still no adversarial
            lpips_weight = 0.0
        else:
            # Only after 20 epochs, start adding adversarial components very cautiously
            progress_ratio = min((epoch - warmup_period - 10) / (num_epochs * 0.5), 1.0)
            content_weight = 1.0
            perceptual_weight = 0.01 + 0.04 * progress_ratio  # Slower increase
            adversarial_weight = 0.00001 * progress_ratio     # Extremely cautious
            lpips_weight = 0.005 * progress_ratio if use_lpips else 0.0
        
        # Create a progress bar for better visibility
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
            batch_count += 1
            
            # Move images to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Check for NaN values in input before processing
            if torch.isnan(lr_imgs).any() or torch.isnan(hr_imgs).any():
                print("NaN values detected in input images, skipping batch")
                continue
                
            # Normalize input to ensure no extreme values
            lr_imgs = torch.clamp(lr_imgs, -1.0, 1.0)
            hr_imgs = torch.clamp(hr_imgs, -1.0, 1.0)
            
            # Adversarial ground truths
            valid = torch.ones((lr_imgs.size(0), 1), device=device)
            fake = torch.zeros((lr_imgs.size(0), 1), device=device)
            
            #-------------------------
            # Train Generator
            #-------------------------
            optimizer_G.zero_grad()
            
            # Use standard torch.cuda.amp.autocast context manager
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                # Generate a high resolution image from low resolution input
                gen_hr = generator(lr_imgs)
                
                # Check for NaN values in generated images
                if torch.isnan(gen_hr).any():
                    print("NaN values detected in generator output, skipping batch")
                    continue
                
                # Adversarial loss
                pred_real = discriminator(hr_imgs)
                pred_fake = discriminator(gen_hr)
                
                # Add label smoothing for discriminator - soften the real labels
                valid_smooth = valid * 0.9
                
                # Calculate generator losses
                gen_adv_loss = adversarial_criterion(pred_fake, valid_smooth)
                
                # Content loss - compare pixels (L1 norm)
                gen_content_loss = content_criterion(gen_hr, hr_imgs)
                
                # Perceptual loss - compare high-level features
                gen_perceptual_loss = vgg_loss(gen_hr, hr_imgs)
                
                # LPIPS loss if available
                if use_lpips:
                    # Convert from [-1, 1] to [0, 1] for LPIPS
                    gen_hr_01 = (gen_hr + 1.0) / 2.0
                    hr_imgs_01 = (hr_imgs + 1.0) / 2.0
                    gen_lpips_loss = lpips_loss(gen_hr_01, hr_imgs_01).mean()
                else:
                    gen_lpips_loss = torch.tensor(0.0, device=device)
                
                # Total generator loss - weighted sum
                gen_loss = (
                    adversarial_weight * gen_adv_loss + 
                    content_weight * gen_content_loss + 
                    perceptual_weight * gen_perceptual_loss + 
                    lpips_weight * gen_lpips_loss
                )
            
            # Use mixed precision gradient scaling
            scaler.scale(gen_loss).backward()
            
            # Apply gradient clipping to prevent extreme weight updates
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            scaler.step(optimizer_G)
            scaler.update()
            
            #-------------------------
            # Train Discriminator
            #-------------------------
            optimizer_D.zero_grad()
            
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                # Compute discriminator outputs
                pred_real = discriminator(hr_imgs)
                pred_fake = discriminator(gen_hr.detach())
                
                # Compute losses with label smoothing - soften the real labels
                real_loss = adversarial_criterion(pred_real, valid_smooth)
                fake_loss = adversarial_criterion(pred_fake, fake)
                
                # Total discriminator loss
                disc_loss = (real_loss + fake_loss) / 2
            
            # Use mixed precision gradient scaling
            scaler.scale(disc_loss).backward()
            
            # Apply gradient clipping to prevent extreme weight updates
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            
            scaler.step(optimizer_D)
            scaler.update()
            
            # Accumulate losses for epoch average
            epoch_gen_adv_loss += gen_adv_loss.item()
            epoch_gen_content_loss += gen_content_loss.item()
            epoch_gen_perceptual_loss += gen_perceptual_loss.item()
            epoch_gen_lpips_loss += gen_lpips_loss.item() if use_lpips else 0
            epoch_disc_loss += disc_loss.item()
            
            # Update progress bar description
            progress_bar.set_postfix({
                "G_loss": f"{gen_loss.item():.4f}",
                "D_loss": f"{disc_loss.item():.4f}"
            })
            
            # Clean up to free memory
            del gen_hr, gen_loss, disc_loss, pred_real, pred_fake
            
        # Calculate average losses for the epoch
        avg_gen_adv_loss = epoch_gen_adv_loss / batch_count
        avg_gen_content_loss = epoch_gen_content_loss / batch_count
        avg_gen_perceptual_loss = epoch_gen_perceptual_loss / batch_count
        avg_gen_lpips_loss = epoch_gen_lpips_loss / batch_count if use_lpips else 0
        avg_disc_loss = epoch_disc_loss / batch_count
        
        # Step the schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Validation phase
        print("\nRunning validation...")
        generator.eval()
        
        # Track validation metrics
        val_content_loss = 0.0
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        val_lpips_total = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for val_lr, val_hr in valid_dataloader:
                val_lr = val_lr.to(device)
                val_hr = val_hr.to(device)
                
                # Generate super-resolution images
                val_sr = generator(val_lr)
                
                # Calculate content loss
                val_batch_content_loss = content_criterion(val_sr, val_hr).item()
                val_content_loss += val_batch_content_loss
                
                # Calculate PSNR
                val_batch_psnr = calculate_psnr(val_sr, val_hr)
                val_psnr_total += val_batch_psnr
                
                # Calculate SSIM
                try:
                    sr_np = (val_sr.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                    hr_np = (val_hr.cpu().numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
                    val_batch_ssim = calculate_ssim(sr_np, hr_np)
                    val_ssim_total += val_batch_ssim
                except Exception as e:
                    print(f"Error calculating SSIM: {e}")
                    val_batch_ssim = 0
                
                # Calculate LPIPS if available
                if use_lpips:
                    try:
                        val_sr_01 = (val_sr + 1.0) / 2.0
                        val_hr_01 = (val_hr + 1.0) / 2.0
                        val_batch_lpips = lpips_loss(val_sr_01, val_hr_01).mean().item()
                        val_lpips_total += val_batch_lpips
                    except Exception as e:
                        print(f"Error calculating LPIPS: {e}")
                        val_batch_lpips = 0
                
                val_batches += 1
        
        # Calculate average validation metrics
        avg_val_content_loss = val_content_loss / max(val_batches, 1)
        avg_val_psnr = val_psnr_total / max(val_batches, 1)
        avg_val_ssim = val_ssim_total / max(val_batches, 1)
        avg_val_lpips = val_lpips_total / max(val_batches, 1) if use_lpips else 0
        
        # Store metrics for plotting later
        metrics['train_g_losses'].append(avg_gen_content_loss)
        metrics['train_d_losses'].append(avg_disc_loss)
        metrics['val_losses'].append(avg_val_content_loss)
        metrics['val_psnr'].append(avg_val_psnr)
        metrics['val_ssim'].append(avg_val_ssim)
        metrics['val_lpips'].append(avg_val_lpips)
        
        # Print validation results
        print(f"Validation - PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}, LPIPS: {avg_val_lpips:.4f}")
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                "train/gen_adv_loss": avg_gen_adv_loss,
                "train/gen_content_loss": avg_gen_content_loss,
                "train/gen_perceptual_loss": avg_gen_perceptual_loss,
                "train/gen_lpips_loss": avg_gen_lpips_loss,
                "train/disc_loss": avg_disc_loss,
                "val/content_loss": avg_val_content_loss,
                "val/psnr": avg_val_psnr,
                "val/ssim": avg_val_ssim,
                "val/lpips": avg_val_lpips,
                "lr/generator": optimizer_G.param_groups[0]['lr'],
                "lr/discriminator": optimizer_D.param_groups[0]['lr'],
            })
            
            # Log sample images
            log_images_to_wandb(generator, valid_dataloader, device, epoch, use_wandb)
        
        # Early stopping logic based on PSNR improvement
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
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
                'epoch': epoch,
            }, "checkpoints/best_model_track2.pth")
            print("Saved best model.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
                break
        
        # Save checkpoint at specified intervals
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'metrics': metrics,
                'best_val_psnr': best_val_psnr,
                'no_improve_epochs': no_improve_epochs
            }, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
    
    # Plot training progress
    plot_training_progress(metrics, save_dir="results_track2/training_plots")
    
    # Load best model for final evaluation
    best_checkpoint = torch.load("checkpoints/best_model_track2.pth")
    generator.load_state_dict(best_checkpoint['generator_state_dict'])
    
    # Save example images with best model
    print("Saving example images with best model...")
    save_example_images(generator, valid_dataset, device, save_dir="results_track2/examples")
    
    # Final evaluation
    print("Final evaluation with best model:")
    generator.eval()
    result = evaluate_model(generator, valid_dataloader, device)
    print(f"Best model - PSNR: {result['psnr']:.2f} dB, SSIM: {result['ssim']:.4f}, LPIPS: {result['lpips']:.4f}")
    
    return generator, discriminator, metrics

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
        ax4.set_ylabel('Difference', fontsize=12)
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

def log_images_to_wandb(generator, valid_loader, device, epoch, use_wandb):
    """Log sample images to wandb for visualization using same samples across epochs"""
    if not use_wandb:
        return
    
    global fixed_validation_samples
    results_dir = "results_track2"
    
    # Set generator to evaluation mode
    generator.eval()
    
    with torch.no_grad():
        # Import necessary libraries for image processing
        from PIL import Image
        import numpy as np
        
        # Special handling for the first time we run the function
        if fixed_validation_samples is None:
            fixed_validation_samples = []
            
            # Use the validation verification images we've already created
            verification_dir = os.path.join(results_dir, 'validation_verification')
            
            # If we've already verified the dataset, use those indices
            if os.path.exists(verification_dir):
                try:
                    # Get the dataset directly
                    dataset = valid_loader.dataset
                    dataset_size = len(dataset)
                    
                    # Use 3 fixed samples from our verification
                    num_samples = 3
                    indices = [i * dataset_size // num_samples for i in range(num_samples)]
                    
                    print(f"Using validated fixed indices: {indices}")
                    
                    # Load the samples directly from the dataset
                    for idx in indices:
                        lr_tensor, hr_tensor = dataset[idx]
                        
                        # Add batch dimension and move to device
                        lr_tensor = lr_tensor.unsqueeze(0).to(device)
                        hr_tensor = hr_tensor.unsqueeze(0).to(device)
                        
                        # Store the paired samples
                        fixed_validation_samples.append({
                            'idx': idx,
                            'lr': lr_tensor,
                            'hr': hr_tensor
                        })
                        
                    print(f"Successfully loaded {len(fixed_validation_samples)} fixed samples")
                except Exception as e:
                    print(f"Error loading fixed samples: {e}")
                    # We'll fall back to the code below
            
            # If we couldn't load verified samples, use the dataloader
            if not fixed_validation_samples:
                print("Using fallback method to load fixed samples")
                try:
                    # Get the first few samples from the dataloader
                    valid_iter = iter(valid_loader)
                    for i in range(3):  # Get 3 samples
                        try:
                            lr_batch, hr_batch = next(valid_iter)
                            fixed_validation_samples.append({
                                'idx': i,
                                'lr': lr_batch.to(device),
                                'hr': hr_batch.to(device)
                            })
                        except StopIteration:
                            break
                        
                except Exception as e:
                    print(f"Error in fallback sample loading: {e}")
            
            # If we still couldn't get any samples, return
            if not fixed_validation_samples:
                print("Failed to load any fixed samples. Skipping visualization.")
                generator.train()
                return
        
        # Process each of our fixed samples with improved error handling
        for i, sample_data in enumerate(fixed_validation_samples):
            try:
                sample_idx = sample_data['idx']
                sample_lr = sample_data['lr']
                sample_hr = sample_data['hr']
                
                # Generate SR image with appropriate error handling
                try:
                    # Ensure the generator is in eval mode
                    generator.eval()
                    
                    # Generate SR image
                    sample_sr = generator(sample_lr)
                    
                    # Check output quality
                    if torch.isnan(sample_sr).any() or torch.isinf(sample_sr).any():
                        print(f"Warning: SR output contains NaN/Inf values for sample {sample_idx}")
                        # Use bicubic upsampling as fallback
                        from torchvision.transforms.functional import resize
                        sample_sr = resize(sample_lr, size=sample_hr.shape[2:], 
                                           mode='bicubic', antialias=True)

                        # Use bicubic upsampling as emergency fallback
                        # Use a more compatible interpolation method
                        sample_sr = torch.nn.functional.interpolate(
                            sample_lr, 
                            size=sample_hr.shape[2:], 
                            mode='bicubic', 
                            align_corners=False
                        )
                except Exception as e:
                    print(f"Error generating SR image: {e}")
                    # Use bicubic upsampling as emergency fallback
                    from torchvision.transforms.functional import resize
                    sample_sr = resize(sample_lr, size=sample_hr.shape[2:], 
                                       mode='bicubic', antialias=True)
                
                # Convert tensors to numpy arrays for visualization
                # Move to CPU and ensure proper range
                sample_lr_np = (sample_lr[0].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                sample_sr_np = (sample_sr[0].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                sample_hr_np = (sample_hr[0].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                
                # Verify array shapes match expected dimensions
                print(f"Image shapes - LR: {sample_lr_np.shape}, SR: {sample_sr_np.shape}, HR: {sample_hr_np.shape}")
                
                # Upscale LR image for display purposes only
                hr_h, hr_w = sample_hr_np.shape[0], sample_hr_np.shape[1]
                lr_pil = Image.fromarray((sample_lr_np * 255).astype(np.uint8))
                lr_upscaled = lr_pil.resize((hr_w, hr_h), Image.BICUBIC)
                lr_upscaled_np = np.array(lr_upscaled) / 255.0
                
                # Create a directory to store the fixed sample images
                sample_dir = os.path.join(results_dir, f"fixed_sample_{i+1}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Create individual images for wandb
                sample_lr_img = wandb.Image(
                    lr_upscaled_np,
                    caption=f"LR Input {i+1} (Upscaled)"
                )
                sample_sr_img = wandb.Image(
                    sample_sr_np,
                    caption=f"SR Output {i+1} (Epoch {epoch+1})"
                )
                sample_hr_img = wandb.Image(
                    sample_hr_np,
                    caption=f"HR Ground Truth {i+1}"
                )
                
                # Create a side-by-side comparison
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(lr_upscaled_np)
                plt.title(f"LR Input {i+1}")
                plt.axis("off")
                
                plt.subplot(1, 3, 2)
                plt.imshow(sample_sr_np)
                plt.title(f"SR Output {i+1} (Epoch {epoch+1})")
                plt.axis("off")
                
                plt.subplot(1, 3, 3)
                plt.imshow(sample_hr_np)
                plt.title(f"HR Ground Truth {i+1}")
                plt.axis("off")
                
                plt.tight_layout()
                
                # Save the comparison
                comparison_path = os.path.join(sample_dir, f"comparison_epoch_{epoch+1}.png")
                plt.savefig(comparison_path, dpi=300)
                plt.close()
                
                # Log the comparison image to wandb
                comparison_img = wandb.Image(
                    comparison_path,
                    caption=f"Comparison {i+1} at Epoch {epoch+1}"
                )
                
                # Create a standardized tag for consistent sample tracking
                sample_tag = f"fixed_sample_{i+1}"
                
                # Log individual and comparison images
                wandb.log({
                    f"images/{sample_tag}/LR_input": sample_lr_img,
                    f"images/{sample_tag}/SR_output": sample_sr_img,
                    f"images/{sample_tag}/HR_ground_truth": sample_hr_img,
                    f"images/{sample_tag}/comparison": comparison_img,
                    "epoch": epoch + 1
                })
                
            except Exception as e:
                print(f"Error processing fixed sample {i+1}: {e}")
    
    # Return generator to training mode
    generator.train()

# Global variables to store fixed samples
fixed_validation_samples = None
fixed_example_indices = None

# Update autocast for CPU/GPU compatibility
def get_autocast():
    """Return the appropriate autocast context manager for the current device"""
    if device.type == 'cuda':
        return torch.cuda.amp.autocast()
    else:
        # For CPU, use torch.amp.autocast with device_type='cpu'
        return torch.amp.autocast(device_type='cpu')

# Function to download DIV2K dataset for Colab
def download_div2k_dataset():
    """Download and extract DIV2K dataset for training and validation"""
    if not IS_COLAB:
        print("This function is only for Google Colab environments")
        return
        
    import os
    import urllib.request
    import zipfile
    
    # Create directory structure
    os.makedirs('DIV2K', exist_ok=True)
    
    # URLs for DIV2K dataset
    urls = {
        'train_HR': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'train_LR_unknown': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip',
        'valid_HR': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        'valid_LR_unknown': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip'
    }
    
    for name, url in urls.items():
        zip_path = f'/content/DIV2K/{name}.zip'
        
        # Only download if not already downloaded
        if not os.path.exists(zip_path):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract zip file
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('/content/DIV2K')
                
            print(f"Done with {name}")
        else:
            print(f"{name} already downloaded")
    
    print("DIV2K dataset download complete")
    
    # Fix folder structure if needed
    if not os.path.exists('/content/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown'):
        os.makedirs('/content/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown', exist_ok=True)
        os.makedirs('/content/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown/X4', exist_ok=True)
        
        # Move files to correct location if necessary
        src_dir = '/content/DIV2K/DIV2K_train_LR_unknown_X4'
        dst_dir = '/content/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown/X4'
        
        import glob
        # Only move image files if they're in the wrong place
        image_files = glob.glob(f"{src_dir}/*.png")
        if image_files:
            import shutil
            for img in image_files:
                shutil.move(img, dst_dir)
    
    # Do the same for validation set
    if not os.path.exists('/content/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown'):
        os.makedirs('/content/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown', exist_ok=True)
        os.makedirs('/content/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown/X4', exist_ok=True)
        
        # Move files to correct location if necessary
        src_dir = '/content/DIV2K/DIV2K_valid_LR_unknown_X4'
        dst_dir = '/content/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown/X4'
        
        import glob
        # Only move image files if they're in the wrong place
        image_files = glob.glob(f"{src_dir}/*.png")
        if image_files:
            import shutil
            for img in image_files:
                shutil.move(img, dst_dir)
    
    # Return updated paths
    return {
        'TRAIN_LR_UNKNOWN_PATH': '/content/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown/X4',
        'TRAIN_HR_PATH': '/content/DIV2K/DIV2K_train_HR',
        'VALID_LR_UNKNOWN_PATH': '/content/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown/X4',
        'VALID_HR_PATH': '/content/DIV2K/DIV2K_valid_HR'
    }

# Function to generate a Colab notebook
def generate_colab_notebook():
    """Generate a Colab notebook to run this script"""
    import json
    
    # Create a Colab notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": "SRGAN_Track2_Training.ipynb",
                "provenance": [],
                "collapsed_sections": [],
                "toc_visible": True,
                "authorship_tag": "SRGAN"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# SRGAN for Unknown Degradation Super-Resolution (Track 2)\n\n",
                          "This notebook runs the SRGAN model for super-resolution on unknown degradation patterns."]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Check if GPU is available\n",
                          "!nvidia-smi"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Clone the repository or upload the script\n",
                           "!git clone https://github.com/YOUR_USERNAME/SRGAN-SuperResolution.git\n",
                           "# If you're not using a repo, upload task2_SRGAN.py manually\n\n",
                           "# Install required packages\n",
                           "!pip install torch torchvision lpips wandb scikit-image matplotlib"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Mount Google Drive\n",
                           "from google.colab import drive\n",
                           "drive.mount('/content/drive')"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Run the training script with dataset download\n",
                           "# Add --use_wandb if you want to use Weights & Biases tracking\n",
                           "!python task2_SRGAN.py --download_dataset"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Visualization and Results\n\n",
                          "After training is complete, you can view the results here."]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Display training plots\n",
                           "import matplotlib.pyplot as plt\n",
                           "from IPython.display import display\n",
                           "import glob\n\n",
                           "# Display training metrics\n",
                           "metrics_plot = glob.glob('/content/results_track2/training_metrics.png')\n",
                           "if metrics_plot:\n",
                           "    img = plt.imread(metrics_plot[0])\n",
                           "    plt.figure(figsize=(15, 10))\n",
                           "    plt.imshow(img)\n",
                           "    plt.axis('off')\n",
                           "    plt.show()"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Display sample results\n",
                           "sample_images = sorted(glob.glob('/content/results_track2/examples/sample_*_comparison.png'))\n",
                           "for img_path in sample_images[:3]:  # Show first 3 samples\n",
                           "    img = plt.imread(img_path)\n",
                           "    plt.figure(figsize=(15, 5))\n",
                           "    plt.imshow(img)\n",
                           "    plt.title(img_path.split('/')[-1])\n",
                           "    plt.axis('off')\n",
                           "    plt.show()"],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Save model and results to Google Drive\n",
                           "!mkdir -p /content/drive/MyDrive/SRGAN_results\n",
                           "!cp -r /content/results_track2/* /content/drive/MyDrive/SRGAN_results/"],
                "execution_count": None,
                "outputs": []
            }
        ]
    }
    
    # Save the notebook
    with open('SRGAN_Track2_Training.ipynb', 'w') as f:
        json.dump(notebook, f)
    
    print("Colab notebook generated: SRGAN_Track2_Training.ipynb")
    print("You can upload this notebook to Google Colab.")
    
    return True

def verify_dataset_pairs(dataset, num_samples=5, save_dir='dataset_verification'):
    """
    Create visual verification of LR and HR pairs from the dataset.
    This is a diagnostic function to ensure proper dataset pairing.
    
    Args:
        dataset: The dataset containing paired LR and HR samples
        num_samples: Number of samples to verify
        save_dir: Directory to save verification images
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Limit samples to dataset size
    num_samples = min(num_samples, len(dataset))
    
    # Choose evenly spaced indices
    indices = [i * len(dataset) // num_samples for i in range(num_samples)]
    
    for i, idx in enumerate(indices):
        # Get a sample pair
        try:
            lr_tensor, hr_tensor = dataset[idx]
            
            # Convert tensors to numpy arrays for visualization
            if isinstance(lr_tensor, torch.Tensor) and isinstance(hr_tensor, torch.Tensor):
                # Handle normalization if needed
                if lr_tensor.min() < 0:  # Assuming [-1, 1] normalization
                    lr_img = (lr_tensor * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    hr_img = (hr_tensor * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                else:  # Assuming [0, 1] normalization
                    lr_img = lr_tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    hr_img = hr_tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                
                # Create bicubic upsampled version of LR for comparison
                lr_pil = Image.fromarray((lr_img * 255).astype(np.uint8))
                hr_h, hr_w = hr_img.shape[0], hr_img.shape[1]
                lr_upscaled = lr_pil.resize((hr_w, hr_h), Image.BICUBIC)
                lr_upscaled_np = np.array(lr_upscaled) / 255.0
                
                # Create a side-by-side comparison
                plt.figure(figsize=(16, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(lr_img)
                plt.title(f"LR (idx {idx})")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(lr_upscaled_np)
                plt.title("LR (Bicubic 4x)")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(hr_img)
                plt.title("HR Ground Truth")
                plt.axis('off')
                
                plt.suptitle(f"Pair {i+1} (Dataset Index {idx})")
                plt.tight_layout()
                
                # Save the comparison
                plt.savefig(os.path.join(save_dir, f"pair_{i+1}_comparison.png"), dpi=300)
                plt.close()
                
                print(f"Saved verification for pair {i+1} (index {idx})")
            else:
                print(f"Error: Sample at index {idx} is not a tensor")
        
        except Exception as e:
            print(f"Error verifying sample at index {idx}: {e}")
    
    print(f"Saved {num_samples} verification images to {save_dir}")
    return save_dir

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SRGAN for Track 2 - Unknown Pairing SR')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--download_dataset', action='store_true', help='Download DIV2K dataset (Colab only)')
    parser.add_argument('--generate_notebook', action='store_true', help='Generate a Colab notebook')
    parser.add_argument('--lr_path', type=str, help='Custom path to LR images directory')
    parser.add_argument('--hr_path', type=str, help='Custom path to HR images directory')
    parser.add_argument('--list_dir', action='store_true', help='List available DIV2K directories')
    args = parser.parse_args()
    
    # Generate notebook if requested
    if args.generate_notebook:
        generate_colab_notebook()
        sys.exit(0)
    
    # List DIV2K directories if requested
    if args.list_dir:
        print("Searching for DIV2K directories...")
        base_path = os.getcwd()
        for root, dirs, files in os.walk(base_path):
            if "DIV2K" in root:
                png_files = [f for f in files if f.lower().endswith(".png")]
                if png_files:
                    print(f"Found directory with images: {root} ({len(png_files)} PNG files)")
        sys.exit(0)
    
    try:
        print("Starting SRGAN training for Track 2 (Unknown Pairing)...")
        print(f"Using synthetic data: {args.use_synthetic}")
        print(f"Using Weights & Biases: {args.use_wandb}")
        
        # Special handling for Colab environment
        if IS_COLAB:
            print("Running in Google Colab environment")
            
            # Setup Colab environment if it's detected
            if 'setup_colab' in globals():
                setup_colab()
            
            # Download dataset if requested or needed
            if args.download_dataset or not os.path.exists(TRAIN_HR_PATH):
                print("Downloading DIV2K dataset...")
                paths = download_div2k_dataset()
                if paths:
                    TRAIN_LR_UNKNOWN_PATH = paths['TRAIN_LR_UNKNOWN_PATH']
                    TRAIN_HR_PATH = paths['TRAIN_HR_PATH']
                    VALID_LR_UNKNOWN_PATH = paths['VALID_LR_UNKNOWN_PATH']
                    VALID_HR_PATH = paths['VALID_HR_PATH']
                    print("Using downloaded dataset paths")
        
        # Verify and update dataset paths
        print("Verifying dataset paths...")
        paths = verify_and_update_dataset_paths(args.lr_path, args.hr_path)
        
        if not paths and not args.use_synthetic:
            print("\nERROR: Could not find valid dataset paths and synthetic data not requested.")
            print("Please specify paths manually using --lr_path and --hr_path arguments")
            print("Or use --list_dir to see available DIV2K directories")
            print("Or use --use_synthetic to train with synthetic data")
            print("Or use --download_dataset to download the DIV2K dataset (works in Colab)")
            sys.exit(1)
        
        # Start training
        train_unknown_srgan(use_synthetic=args.use_synthetic, use_wandb=args.use_wandb)
        
        # Load the best model for evaluation
        try:
            print("\nLoading best model for final evaluation...")
            checkpoint_path = os.path.join("results_track2", "track2_srgan_final.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                
                # Create model and load state
                generator = ImprovedGenerator().to(device)
                generator.load_state_dict(checkpoint['generator_state_dict'])
                
                # Create test dataset for final visualization
                hr_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                
                lr_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                
                test_dataset = PairedUnknownDataset(
                    VALID_LR_UNKNOWN_PATH, VALID_HR_PATH,
                    lr_transform=lr_transform,
                    hr_transform=hr_transform
                )
                
                # Generate final samples
                save_dir = os.path.join("results_track2", "final_results")
                save_compact_comparison(generator, test_dataset, device, save_dir=save_dir, num_images=5)
                print(f"Final evaluation images saved to {save_dir}")
                
                # Print the metrics from the checkpoint
                if 'test_metrics' in checkpoint:
                    metrics = checkpoint['test_metrics']
                    print("\nFinal Model Metrics:")
                    print(f"PSNR: {metrics['psnr']:.2f} dB")
                    print(f"SSIM: {metrics['ssim']:.4f}")
                    print(f"LPIPS: {metrics['lpips']:.4f}")
            else:
                print(f"Could not find best model at {checkpoint_path}")
        
        except Exception as e:
            print(f"Error during final evaluation: {e}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
