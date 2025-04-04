import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from math import log10
import matplotlib.pyplot as plt

# -------------------------------------
# Device configuration
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------
# Folder Paths for Unknown Degradation (Track 2)
# -------------------------------------
TRAIN_LR_UNKNOWN_PATH = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_train_LR_unknown_X4/DIV2K_train_LR_unknown/X4"
TRAIN_HR_PATH         = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_train_HR"
VALID_LR_UNKNOWN_PATH = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_valid_LR_unknown_X4/DIV2K_valid_LR_unknown/X4"
VALID_HR_PATH         = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR"

# -------------------------------------
# Dataset Class for Paired Unknown Data (Track 2)
# -------------------------------------
class PairedUnknownDataset(Dataset):
    """
    Loads paired LR (unknown degradation) and HR images from specified directories.
    Assumes matching filenames in both LR and HR folders.
    """
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        n_files = min(len(self.lr_files), len(self.hr_files))
        self.lr_files = self.lr_files[:n_files]
        self.hr_files = self.hr_files[:n_files]
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        if self.lr_transform:
            lr_img = self.lr_transform(lr_img)
        if self.hr_transform:
            hr_img = self.hr_transform(hr_img)
        return lr_img, hr_img

# -------------------------------------
# SRCNN Model for Track 2
# -------------------------------------
class SRCNN(nn.Module):
    """
    SRCNN Model for image super-resolution.
    Architecture:
      - Layer 1: Convolution with 64 filters (9x9 kernel) and ReLU activation.
      - Layer 2: Convolution with 32 filters (5x5 kernel) and ReLU activation.
      - Layer 3: Convolution with 3 filters (5x5 kernel) to output the HR image.
    """
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x

def calculate_psnr(sr, hr):
    mse = nn.MSELoss()(sr, hr).item()
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)

# -------------------------------------
# Training Function for Track 2 using SRCNN
# -------------------------------------
def train_srcnn_track2():
    print("Starting training for Track 2 (SRCNN with Unknown LR images)")
    
    # Define transforms:
    # HR images: Center crop to 256x256, then convert to tensor and normalize.
    hr_transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # LR images: Center crop to 64x64, then upscale to 256x256 using bicubic interpolation,
    # then convert to tensor and normalize.
    lr_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create training and validation datasets using the paired unknown dataset class
    train_dataset = PairedUnknownDataset(TRAIN_LR_UNKNOWN_PATH, TRAIN_HR_PATH,
                                         lr_transform=lr_transform,
                                         hr_transform=hr_transform)
    valid_dataset = PairedUnknownDataset(VALID_LR_UNKNOWN_PATH, VALID_HR_PATH,
                                         lr_transform=lr_transform,
                                         hr_transform=hr_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Initialize SRCNN model
    model = SRCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 50  # Adjust number of epochs as needed
    train_losses = []
    validation_psnrs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_loss:.4f}")
        
        model.eval()
        cumulative_psnr = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                sr_imgs = model(lr_imgs)
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                cumulative_psnr += psnr
        avg_psnr = cumulative_psnr / len(valid_loader)
        validation_psnrs.append(avg_psnr)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation PSNR: {avg_psnr:.2f} dB")
    
    # Plot training curves
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, marker='o', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, validation_psnrs, marker='o', color='green', label='Validation PSNR')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("track2_srcnn_performance.png")
    plt.show()
    
    # Save the trained model
    torch.save(model.state_dict(), "track2_srcnn_model.pth")
    print("Training complete. Model saved as 'track2_srcnn_model.pth'.")

if __name__ == "__main__":
    train_srcnn_track2()
