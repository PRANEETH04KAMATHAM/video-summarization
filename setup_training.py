#!/usr/bin/env python3
"""
Setup script for training environment
"""

import os
import subprocess
import sys

def install_requirements():
    """Install additional requirements for training"""
    training_requirements = [
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2"
    ]
    
    for req in training_requirements:
        print(f"Installing {req}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])

def create_directories():
    """Create necessary directories"""
    dirs = [
        'data/datasets/tvsum/video',
        'data/datasets/tvsum/data',
        'data/datasets/processed',
        'models',
        'results/training'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_sample_data():
    """Download sample data for testing"""
    print("\nDownloading sample dataset...")
    
    # Create sample annotation data
    import json
    sample_annotations = {
        'sample_video_1': {
            'category': 'tutorial',
            'length': 120.0,
            'gt_scores': [0.1] * 50 + [0.8] * 10 + [0.2] * 40  # 100 frames
        }
    }
    
    with open('data/datasets/processed/sample_annotations.json', 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print("Sample annotations created.")

def main():
    print("Setting up training environment...")
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Download sample data
    download_sample_data()
    
    print("\nTraining environment setup complete!")
    print("\nNext steps:")
    print("1. Download TVSum dataset to data/datasets/tvsum/")
    print("2. Run: python train_model.py --prepare_data")
    print("3. Run: python train_model.py")
    print("4. Run: python main.py")

if __name__ == "__main__":
    main()
