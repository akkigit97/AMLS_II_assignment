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
    Assumes that filenames match in both the LR and HR folders.
    """
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        # List image files from each folder (adjust extensions if needed)
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Use only the minimum number of files for pairing
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
# Model Components: Residual Block, Generator, Discriminator
# -------------------------------------
class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.prelu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual

class Generator(nn.Module):
    """
    Generator network for SRGAN (Track 2).
    Upsamples LR images by a factor of 4 to produce HR images.
    Expected input size: upscaled unknown LR image of size 256x256.
    (The LR transform upscales a cropped 64x64 image to 256x256 via bicubic interpolation.)
    """
    def __init__(self, num_residual_blocks=4):
        super(Generator, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        # Upsampling: Here, since our LR transform already upscales to 256x256,
        # we do not need additional upsampling layers.
        self.final_conv = nn.Conv2d(64, 3, kernel_size=5, padding=2)

    def forward(self, x):
        out1 = self.prelu(self.initial_conv(x))
        out = self.res_blocks(out1)
        out = self.bn(self.mid_conv(out))
        out = out1 + out  # Skip connection
        out = self.final_conv(out)
        return torch.tanh(out)  # Output normalized to [-1, 1]

class Discriminator(nn.Module):
    """
    Discriminator network for SRGAN (Track 2).
    Processes HR images (256x256) and outputs a scalar "realness" score.
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        in_channels, height, width = input_shape
        def disc_block(in_filters, out_filters, stride):
            return [
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *disc_block(64, 64, stride=2),
            *disc_block(64, 128, stride=1),
            *disc_block(128, 128, stride=2),
            *disc_block(128, 256, stride=1),
            *disc_block(256, 256, stride=2),
            *disc_block(256, 512, stride=1),
            *disc_block(512, 512, stride=2)
        )
        ds_size = height // 16  # Downsampled size after convolution blocks
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
        
    def forward(self, img):
        features = self.model(img)
        features = features.view(features.size(0), -1)
        validity = self.adv_layer(features)
        return validity

def calculate_psnr(sr, hr):
    mse = nn.MSELoss()(sr, hr).item()
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)

# -------------------------------------
# Training Function for Track 2 using SRGAN
# -------------------------------------
def train_unknown_srgan():
    print("Starting training for Track 2 (SRGAN with Unknown LR images)")
    
    # Define transforms:
    # HR transform: center crop to 256x256, then convert to tensor and normalize.
    hr_transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    # LR transform: center crop to 64x64, then upscale to 256x256 using bicubic interpolation,
    # then convert to tensor and normalize.
    lr_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # Create paired dataset for unknown degradation
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
    
    # Initialize models
    generator = Generator(num_residual_blocks=4).to(device)
    discriminator = Discriminator(input_shape=(3, 256, 256)).to(device)
    
    # Loss functions
    content_loss_fn = nn.MSELoss().to(device)
    adversarial_loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    # Adversarial labels
    real_label = 1.0
    fake_label = 0.0
    
    # Tracking lists
    generator_losses = []
    discriminator_losses = []
    validation_psnrs = []
    
    num_epochs = 50  # Adjust the number of epochs as needed
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_G_loss = 0.0
        total_D_loss = 0.0
        
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_outputs = discriminator(hr_imgs)
            real_labels = torch.full((hr_imgs.size(0), 1), real_label, device=device)
            loss_real = adversarial_loss_fn(real_outputs, real_labels)
            
            fake_imgs = generator(lr_imgs)
            fake_outputs = discriminator(fake_imgs.detach())
            fake_labels = torch.full((hr_imgs.size(0), 1), fake_label, device=device)
            loss_fake = adversarial_loss_fn(fake_outputs, fake_labels)
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_imgs = generator(lr_imgs)
            fake_outputs = discriminator(fake_imgs)
            loss_G_adv = adversarial_loss_fn(fake_outputs, real_labels)
            loss_G_content = content_loss_fn(fake_imgs, hr_imgs)
            loss_G = loss_G_content + 1e-3 * loss_G_adv
            loss_G.backward()
            optimizer_G.step()
            
            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()
        
        avg_G_loss = total_G_loss / len(train_loader)
        avg_D_loss = total_D_loss / len(train_loader)
        generator_losses.append(avg_G_loss)
        discriminator_losses.append(avg_D_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Generator Loss: {avg_G_loss:.4f}, Discriminator Loss: {avg_D_loss:.4f}")
        
        # Validation: Compute PSNR on validation set
        generator.eval()
        cumulative_psnr = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                sr_imgs = generator(lr_imgs)
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                cumulative_psnr += psnr
        avg_psnr = cumulative_psnr / len(valid_loader)
        validation_psnrs.append(avg_psnr)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation PSNR: {avg_psnr:.2f} dB")
    
    # Plot training curves
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, generator_losses, marker='o', label='Generator Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, discriminator_losses, marker='o', color='orange', label='Discriminator Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, validation_psnrs, marker='o', color='green', label='Validation PSNR')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("track2_srgan_performance.png")
    plt.show()
    
    # Save the trained generator model
    torch.save(generator.state_dict(), "track2_srgan_generator.pth")
    print("Training complete. Generator model saved as 'track2_srgan_generator.pth'.")

if __name__ == "__main__":
    train_unknown_srgan()
