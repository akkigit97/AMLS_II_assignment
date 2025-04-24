#!/usr/bin/env python3
"""
Main.py for SRGAN tasks.
This script invokes Track 1.py for Track1 Bicubic and Track 2.py for Track2 unknown degradation.
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main orchestrator for SRGAN tasks.')
    parser.add_argument(
        '--task1', action='store_true', help='Run SRGAN Track1 training (Track 1.py)')
    parser.add_argument(
        '--task2', action='store_true', help='Run SRGAN Track2 training (Track 2.py)')
    parser.add_argument(
        '--use_synthetic', action='store_true', help='Use synthetic data (affects Track2)')
    parser.add_argument(
        '--use_wandb', action='store_true', help='Enable Weights & Biases logging (Track2)')
    parser.add_argument(
        '--download_dataset', action='store_true', help='Download DIV2K dataset (Track2 only)')
    parser.add_argument(
        '--generate_notebook', action='store_true', help='Generate Colab notebook (Track2 only)')
    parser.add_argument(
        '--lr_path', type=str, help='Custom path to LR images (Track2 only)')
    parser.add_argument(
        '--hr_path', type=str, help='Custom path to HR images (Track2 only)')
    parser.add_argument(
        '--norm_type', type=str, choices=['batch','instance','pixel'], default='batch', help='Normalization type for residual blocks (Track2 only)')
    return parser.parse_args()


def run_task1():
    cmd = [sys.executable, 'Track 1.py']
    print(f"Running Track1: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Task1 (Track1) failed with exit code", result.returncode)
        sys.exit(result.returncode)


def run_task2(args):
    cmd = [sys.executable, 'Track 2.py']
    if args.use_synthetic:
        cmd.append('--use_synthetic')
    if args.use_wandb:
        cmd.append('--use_wandb')
    if args.download_dataset:
        cmd.append('--download_dataset')
    if args.generate_notebook:
        cmd.append('--generate_notebook')
    if args.lr_path:
        cmd.extend(['--lr_path', args.lr_path])
    if args.hr_path:
        cmd.extend(['--hr_path', args.hr_path])
    if hasattr(args, 'norm_type') and args.norm_type:
        cmd.extend(['--norm_type', args.norm_type])

    print(f"Running Track2: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Task2 (Track2) failed with exit code", result.returncode)
        sys.exit(result.returncode)


def main():
    args = parse_args()
    # Default: run both if no specific task flag is provided
    run1 = args.task1 or (not args.task1 and not args.task2)
    run2 = args.task2 or (not args.task1 and not args.task2)

    if run1:
        run_task1()
    if run2:
        run_task2(args)

    print("All requested tasks completed successfully.")


if __name__ == '__main__':
main()
