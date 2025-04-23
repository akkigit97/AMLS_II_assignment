# Super-Resolution GAN (SRGAN) Implementation

This repository contains two implementations of SRGAN (Super-Resolution Generative Adversarial Network) for image super-resolution tasks:

1. `Track 1.py`: Enhanced SRGAN for standard super-resolution with known degradation patterns - bicubic (Track 1)
2. `Track 2.py`: Specialized SRGAN for unknown degradation super-resolution (Track 2)


### Prerequisites

Install the required packages:

```bash
pip install torch torchvision lpips wandb scikit-image matplotlib
```

### main.py

This repository includes a top-level script `main.py` that will invoke both Track1 and Track2 training by default. You can use the following commands:

```bash
# Run both Track1 and Track2 in sequence (default)
python main.py

# Run only the Track1 script (Track 1.py)
python main.py --task1

# Run only the Track2 script with SRGAN-Track2 options
python main.py --task2 [--use_synthetic] [--use_wandb] [--download_dataset] [--generate_notebook] [--lr_path LR_DIR] [--hr_path HR_DIR]

# View help for all options
tools: python main.py --help
```

### Direct Script Execution (Legacy)

You can still run each task script directly:

```bash
# Task1: Standard super-resolution
python "Track 1.py"

# Task2: Unknown degradation super-resolution
python "Track 2.py" --use_wandb
```

### Command Line Arguments (Track2)

- `--use_synthetic`: Use synthetic data for quick tests
- `--use_wandb`: Enable Weights & Biases logging
- `--download_dataset`: Download DIV2K dataset (Colab only)
- `--generate_notebook`: Generate a Colab notebook template
- `--lr_path PATH`: Custom path to low-resolution images
- `--hr_path PATH`: Custom path to high-resolution images

## Memory Management

Both implementations include memory optimization techniques:

1. Explicit GPU cache clearing
2. Garbage collection between critical operations
3. Tensor cleanup in validation/test loops
4. Adaptive batch size based on available memory
5. Gradient accumulation for memory-efficient training

## Visualization Tools

The implementation provides:

1. Training metrics visualization with matplotlib
2. Sample image generation and comparison
3. Wandb integration for online experiment tracking
4. PSNR/SSIM/LPIPS metric logging

## Note on Cross-platform Compatibility

Track 2.py includes specific adaptations for Windows compatibility:

1. Resource limit handling with fallback mechanisms
2. Path handling using os.path.join for cross-platform paths
3. Error handling for Windows-specific issues