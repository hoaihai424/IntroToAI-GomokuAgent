#!/usr/bin/env python3
"""
Training script for sklearn-based Gomoku agent.
Trains MLPClassifier on generated game data.
"""

import argparse
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from training.sklearn_models import (
    prepare_training_data,
    create_mlp_model,
    save_sklearn_model
)
from training.dataset_utils import load_dataset


def main():
    parser = argparse.ArgumentParser(description='Train sklearn MLP for Gomoku')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (.npz)')
    parser.add_argument('--output', type=str, default='checkpoints/sklearn_model.pkl',
                       help='Output model path')
    parser.add_argument('--hidden-layers', type=str, default='512,256,128',
                       help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--max-iter', type=int, default=500,
                       help='Maximum training iterations')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test set fraction')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation set fraction')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--filter-winning', action='store_true',
                       help='Only use winning moves for training')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SKLEARN MLP TRAINING FOR GOMOKU")
    print("="*70)
    
    # Parse hidden layers
    hidden_layers = tuple(int(x) for x in args.hidden_layers.split(','))
    
    print(f"\nConfiguration:")
    print(f"  Data file:        {args.data}")
    print(f"  Output model:     {args.output}")
    print(f"  Hidden layers:    {hidden_layers}")
    print(f"  Max iterations:   {args.max_iter}")
    print(f"  Test split:       {args.test_split}")
    print(f"  Val split:        {args.val_split}")
    print(f"  Filter winning:   {args.filter_winning}")
    print(f"  Random seed:      {args.random_seed}")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading dataset...")
    boards, moves, players, outcomes, metadata = load_dataset(args.data)
    
    print(f"\nDataset statistics:")
    print(f"  Total samples:    {len(boards):,}")
    print(f"  Player 1 samples: {np.sum(players == 1):,}")
    print(f"  Player 2 samples: {np.sum(players == 2):,}")
    print(f"  Winning moves:    {np.sum(outcomes == 1):,}")
    print(f"  Losing moves:     {np.sum(outcomes == 0):,}")
    
    # Prepare training data
    print("\n🔄 Preparing training data...")
    X, y = prepare_training_data(boards, moves, players, outcomes, 
                                 filter_winning=args.filter_winning)
    
    print(f"  Feature shape:    {X.shape}")
    print(f"  Label shape:      {y.shape}")
    print(f"  Label range:      [{y.min()}, {y.max()}]")
    
    # Split data
    print(f"\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.random_seed, shuffle=True
    )
    
    val_size = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=args.random_seed, shuffle=True
    )
    
    print(f"  Train: {len(X_train):,} samples ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Val:   {len(X_val):,} samples ({100*len(X_val)/len(X):.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({100*len(X_test)/len(X):.1f}%)")
    
    # Create model
    print(f"\n🧠 Creating MLP model...")
    model = create_mlp_model(
        hidden_layers=hidden_layers,
        max_iter=args.max_iter,
        random_state=args.random_seed
    )
    
    print(f"  Architecture: 675 → {' → '.join(map(str, hidden_layers))} → 225")
    total_params = 675 * hidden_layers[0]
    for i in range(len(hidden_layers) - 1):
        total_params += hidden_layers[i] * hidden_layers[i+1]
    total_params += hidden_layers[-1] * 225
    print(f"  Total parameters: ~{total_params:,}")
    
    # Train model
    print(f"\n🚀 Training model...")
    print("="*70)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print("="*70)
    print(f"✓ Training complete in {training_time/60:.2f} minutes")
    
    # Evaluate model
    print(f"\n📊 Evaluating model...")
    
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f} ({100*train_acc:.2f}%)")
    print(f"  Val:   {val_acc:.4f} ({100*val_acc:.2f}%)")
    print(f"  Test:  {test_acc:.4f} ({100*test_acc:.2f}%)")
    
    # Top-K accuracy
    print(f"\nTop-K Accuracy (Test set):")
    y_pred_proba = model.predict_proba(X_test)
    for k in [1, 3, 5, 10]:
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        top_k_acc = np.mean([y_test[i] in top_k_preds[i] for i in range(len(y_test))])
        print(f"  Top-{k:2d}: {top_k_acc:.4f} ({100*top_k_acc:.2f}%)")
    
    # Save model
    print(f"\n💾 Saving model...")
    
    metadata = {
        'hidden_layers': list(hidden_layers),
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'training_time_minutes': training_time / 60,
        'training_samples': len(X_train),
        'iterations': int(model.n_iter_),
        'final_loss': float(model.loss_),
        'data_file': args.data,
        'filter_winning': args.filter_winning
    }
    
    save_sklearn_model(model, args.output, metadata)
    
    # Summary
    print(f"\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel saved to: {args.output}")
    print(f"Test accuracy: {100*test_acc:.2f}%")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"\nNext steps:")
    print(f"  1. Test model: python test_sklearn_agent.py --model {args.output}")
    print(f"  2. Play in GUI: python play_gomoku.py")
    print()


if __name__ == "__main__":
    main()
