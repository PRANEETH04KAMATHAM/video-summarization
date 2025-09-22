#!/usr/bin/env python3
"""
Training script for video summarization model
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
from data_preparation import TVSumDataPreprocessor
from feature_extractor import LSTMVideoSummarizer
import numpy as np

def plot_training_history(history, save_path="results/training_history.png"):
    """Plot training history"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(history.history['mean_absolute_error'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history saved to: {save_path}")

def evaluate_model(model, test_data):
    """Evaluate model performance"""
    print("Evaluating model...")
    
    total_mae = 0
    total_samples = 0
    
    for sample in test_data:
        features = sample['features']
        gt_scores = sample['gt_scores']
        
        # Predict scores
        pred_scores = model.predict_importance_scores(features)
        
        # Calculate MAE
        mae = np.mean(np.abs(pred_scores - gt_scores))
        total_mae += mae
        total_samples += 1
    
    avg_mae = total_mae / total_samples
    print(f"Average MAE: {avg_mae:.4f}")
    
    return avg_mae

def main():
    parser = argparse.ArgumentParser(description='Train video summarization model')
    parser.add_argument('--dataset_path', default='data/datasets', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare training data')
    args = parser.parse_args()
    
    print("="*60)
    print("VIDEO SUMMARIZATION MODEL TRAINING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = TVSumDataPreprocessor(args.dataset_path)
    
    # Step 1: Prepare data if needed
    if args.prepare_data:
        print("\n1. PREPARING TRAINING DATA")
        print("-" * 30)
        
        # Setup TVSum dataset
        tvsum_path = preprocessor.download_and_setup_tvsum()
        
        # Check if dataset exists
        if not os.path.exists(os.path.join(tvsum_path, "video")):
            print("Please download the TVSum dataset first!")
            print("Follow the instructions printed above.")
            return
        
        # Prepare training data
        training_data = preprocessor.prepare_training_data(tvsum_path)
        
        if not training_data:
            print("No training data found! Please check your dataset setup.")
            return
        
        # Save processed data
        data_file = preprocessor.save_processed_data(training_data)
        print(f"Training data prepared and saved to: {data_file}")
    
    # Step 2: Load training data
    print("\n2. LOADING TRAINING DATA")
    print("-" * 30)
    
    try:
        training_data = preprocessor.load_processed_data()
        print(f"Loaded {len(training_data)} training samples")
    except FileNotFoundError:
        print("Training data not found! Run with --prepare_data first.")
        return
    
    # Step 3: Split data
    print("\n3. SPLITTING DATA")
    print("-" * 30)
    
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Step 4: Initialize and train model
    print("\n4. TRAINING MODEL")
    print("-" * 30)
    
    # Create model
    model = LSTMVideoSummarizer(input_dim=2048, hidden_dim=256)
    
    # Train model
    history = model.train(
        training_data=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 5: Evaluate model
    print("\n5. EVALUATING MODEL")
    print("-" * 30)
    
    mae = evaluate_model(model, test_data)
    
    # Step 6: Save results
    print("\n6. SAVING RESULTS")
    print("-" * 30)
    
    # Save model
    model_path = "models/trained_lstm_summarizer.h5"
    model.save_model(model_path)
    
    # Plot training history
    plot_training_history(history)
    
    # Save training results
    results = {
        'training_samples': len(train_data),
        'test_samples': len(test_data),
        'final_mae': float(mae),
        'epochs_trained': len(history.history['loss']),
        'model_path': model_path
    }
    
    results_file = "results/training_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training results saved to: {results_file}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved to: {model_path}")
    print(f"Final MAE: {mae:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
