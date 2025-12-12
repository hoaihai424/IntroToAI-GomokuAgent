"""
Sklearn-specific utilities for Gomoku ML training.
Provides preprocessing, model creation, and helper functions.
"""

import numpy as np
import joblib
import os
import json
from typing import Tuple, Optional, Dict
from sklearn.neural_network import MLPClassifier


def board_to_features(board_grid: np.ndarray, current_player: int) -> np.ndarray:
    """
    Convert board state to 675-dimensional feature vector.
    
    Args:
        board_grid: 15×15 board array (0=empty, 1=player1, 2=player2)
        current_player: Current player (1 or 2)
        
    Returns:
        675-dimensional numpy array (3 channels × 15 × 15 flattened)
    """
    # Create 3 channels
    channel_my_stones = (board_grid == current_player).astype(np.float32)
    channel_opp_stones = (board_grid == (3 - current_player)).astype(np.float32)
    channel_empty = (board_grid == 0).astype(np.float32)
    
    # Stack and flatten
    features_3d = np.stack([channel_my_stones, channel_opp_stones, channel_empty], axis=0)
    features_1d = features_3d.flatten()
    
    return features_1d


def move_to_class(row: int, col: int, board_size: int = 15) -> int:
    """Convert (row, col) position to class index 0-224."""
    return row * board_size + col


def class_to_move(class_index: int, board_size: int = 15) -> Tuple[int, int]:
    """Convert class index 0-224 to (row, col) position."""
    row = class_index // board_size
    col = class_index % board_size
    return (row, col)


def prepare_training_data(
    boards: np.ndarray,
    moves: np.ndarray,
    players: np.ndarray,
    outcomes: np.ndarray,
    filter_winning: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw dataset to sklearn format.
    
    Args:
        boards: Board states (N, 15, 15)
        moves: Move positions (N, 2)
        players: Current player for each state (N,)
        outcomes: Game outcomes (N,) - 1 if won, 0 if lost
        filter_winning: If True, only keep winning moves
        
    Returns:
        X (N, 675): Feature vectors
        y (N,): Class labels 0-224
    """
    # Filter data if requested
    if filter_winning:
        mask = outcomes == 1
        boards = boards[mask]
        moves = moves[mask]
        players = players[mask]
        outcomes = outcomes[mask]
        print(f"  Filtered to {len(boards)} winning moves")
    
    # Prepare features and labels
    X = []
    y = []
    
    print(f"  Converting {len(boards)} samples to features...")
    for i in range(len(boards)):
        # Convert board to features
        features = board_to_features(boards[i], players[i])
        X.append(features)
        
        # Convert move to class label
        label = move_to_class(moves[i][0], moves[i][1])
        y.append(label)
        
        # Progress indicator
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{len(boards)} samples...")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    return X, y


def create_mlp_model(
    hidden_layers: Tuple[int, ...] = (512, 256, 128),
    max_iter: int = 500,
    random_state: int = 42
) -> MLPClassifier:
    """
    Create and configure MLPClassifier for Gomoku.
    
    Args:
        hidden_layers: Tuple of hidden layer sizes
        max_iter: Maximum training iterations
        random_state: Random seed
        
    Returns:
        Configured MLPClassifier
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,  # Patience
        verbose=True,
        warm_start=False
    )
    
    return model


def save_sklearn_model(
    model: MLPClassifier,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """
    Save trained sklearn model to disk.
    
    Args:
        model: Trained MLPClassifier
        filepath: Save path (.pkl or .joblib)
        metadata: Optional metadata dictionary
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")
    
    # Save metadata
    if metadata:
        meta_path = filepath.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {meta_path}")


def load_sklearn_model(filepath: str) -> MLPClassifier:
    """
    Load trained sklearn model from disk.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded MLPClassifier
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    
    return model
