import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from math import log10
import numpy as np
import matplotlib.pyplot as plt  # For plotting graphs

#####################
# Folder Paths
#####################
TRAIN_LR_PATH = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4"
TRAIN_HR_PATH = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_train_HR"
VALID_LR_PATH = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4"
VALID_HR_PATH = "/home/uceeaat/AMLS_II_assignment/DIV2K/DIV2K_valid_HR"


#####################
# Dataset
#####################
class DIV2KDataset(Dataset):
    """
    Loads LR-HR image pairs from the DIV2K dataset folders.
    Assumes matching filenames in LR and HR directories (e.g., 0001.png in both).
    """
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        # Filter only valid image files (adjust extensions as needed)
        self.lr_files = [f for f in sorted(os.listdir(lr_dir))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
                         os.path.isfile(os.path.join(lr_dir, f))]
        self.hr_files = [f for f in sorted(os.listdir(hr_dir))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
                         os.path.isfile(os.path.join(hr_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img

#####################
# Simple SRCNN Model
#####################
class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        # Basic 3-layer SRCNN architecture
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

#####################
# PSNR Calculation
#####################
def calculate_psnr(sr, hr):
    """
    sr, hr: Tensors in [0,1] with shape [C,H,W]
    Returns PSNR in dB.
    """
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)  # Assuming images normalized to [0,1]

#####################
# Main Training Loop
#####################
def main():
    # 1. Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Create datasets & loaders with a joint transform (center crop)
    crop_size = 256  # Adjust as needed
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

    train_dataset = DIV2KDataset(TRAIN_LR_PATH, TRAIN_HR_PATH, transform=transform)
    valid_dataset = DIV2KDataset(VALID_LR_PATH, VALID_HR_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")

    # 3. Initialize model, loss, optimizer
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Lists to store metrics for plotting later
    train_losses = []
    val_psnrs = []

    # 4. Train
    num_epochs = 50  # Increase for real training (e.g., 50+ epochs)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)  # Super-resolved output
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}/{num_epochs}] Training Loss: {avg_loss:.4f}")

        # 5. Validate
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = model(lr_imgs)
                val_psnr += calculate_psnr(sr_imgs, hr_imgs)

        avg_psnr = val_psnr / len(valid_loader)
        val_psnrs.append(avg_psnr)
        print(f"[Epoch {epoch+1}/{num_epochs}] Validation PSNR: {avg_psnr:.2f} dB\n")

    # 6. (Optional) Save the model
    torch.save(model.state_dict(), "srcnn_bicubic_x4.pth")
    print("Training complete. Model saved to srcnn_bicubic_x4.pth")

    # 7. Plot the performance graphs
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # Plot Validation PSNR
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_psnrs, marker='o', linestyle='-')
    plt.title("Validation PSNR over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    
    plt.tight_layout()
    plt.savefig("training_performance.png")  # Save the plot to a file
    plt.show()

if __name__ == "__main__":
    main()
