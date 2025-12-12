"""
Dataset utilities for saving, loading, and managing training data.
"""

import numpy as np
import os
from typing import Tuple, Dict, Optional
import json


def save_dataset(
    boards: np.ndarray,
    moves: np.ndarray,
    players: np.ndarray,
    outcomes: np.ndarray,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """
    Save training dataset to disk.
    
    Args:
        boards: Board states (N, H, W)
        moves: Move positions (N, 2) as (row, col)
        players: Current player for each state (N,)
        outcomes: Game outcomes (N,) - 1 if player won, 0 if lost/draw
        filepath: Path to save file (.npz)
        metadata: Optional metadata dictionary
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save arrays
    np.savez_compressed(
        filepath,
        boards=boards,
        moves=moves,
        players=players,
        outcomes=outcomes
    )
    
    # Save metadata separately as JSON
    if metadata:
        meta_filepath = filepath.replace('.npz', '_metadata.json')
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved {len(boards)} samples to {filepath}")
    print(f"  Board shape: {boards.shape}")
    print(f"  Moves shape: {moves.shape}")
    print(f"  File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Load training dataset from disk.
    
    Args:
        filepath: Path to dataset file (.npz)
        
    Returns:
        Tuple of (boards, moves, players, outcomes, metadata)
    """
    # Load arrays
    data = np.load(filepath)
    boards = data['boards']
    moves = data['moves']
    players = data['players']
    outcomes = data['outcomes']
    
    # Load metadata if exists
    metadata = None
    meta_filepath = filepath.replace('.npz', '_metadata.json')
    if os.path.exists(meta_filepath):
        with open(meta_filepath, 'r') as f:
            metadata = json.load(f)
    
    print(f"✓ Loaded {len(boards)} samples from {filepath}")
    print(f"  Board shape: {boards.shape}")
    print(f"  Moves shape: {moves.shape}")
    
    return boards, moves, players, outcomes, metadata


def split_dataset(
    boards: np.ndarray,
    moves: np.ndarray,
    players: np.ndarray,
    outcomes: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        boards: Board states
        moves: Move positions
        players: Players
        outcomes: Game outcomes
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data (rest becomes test)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data) where each is
        (boards, moves, players, outcomes)
    """
    n_samples = len(boards)
    
    # Create indices
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Split data
    train_data = (
        boards[train_idx],
        moves[train_idx],
        players[train_idx],
        outcomes[train_idx]
    )
    
    val_data = (
        boards[val_idx],
        moves[val_idx],
        players[val_idx],
        outcomes[val_idx]
    )
    
    test_data = (
        boards[test_idx],
        moves[test_idx],
        players[test_idx],
        outcomes[test_idx]
    )
    
    print(f"✓ Dataset split:")
    print(f"  Training:   {len(train_idx):6d} samples ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"  Validation: {len(val_idx):6d} samples ({len(val_idx)/n_samples*100:.1f}%)")
    print(f"  Test:       {len(test_idx):6d} samples ({len(test_idx)/n_samples*100:.1f}%)")
    
    return train_data, val_data, test_data


def combine_datasets(
    *datasets: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine multiple datasets into one.
    
    Args:
        *datasets: Variable number of (boards, moves, players, outcomes) tuples
        
    Returns:
        Combined (boards, moves, players, outcomes)
    """
    all_boards = []
    all_moves = []
    all_players = []
    all_outcomes = []
    
    for boards, moves, players, outcomes in datasets:
        all_boards.append(boards)
        all_moves.append(moves)
        all_players.append(players)
        all_outcomes.append(outcomes)
    
    combined = (
        np.concatenate(all_boards),
        np.concatenate(all_moves),
        np.concatenate(all_players),
        np.concatenate(all_outcomes)
    )
    
    print(f"✓ Combined {len(datasets)} datasets into {len(combined[0])} samples")
    
    return combined


def get_dataset_stats(
    boards: np.ndarray,
    moves: np.ndarray,
    players: np.ndarray,
    outcomes: np.ndarray
) -> Dict:
    """
    Get statistics about a dataset.
    
    Args:
        boards: Board states
        moves: Move positions
        players: Players
        outcomes: Game outcomes
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_samples': len(boards),
        'board_shape': boards.shape,
        'unique_boards': len(np.unique(boards.reshape(len(boards), -1), axis=0)),
        'player_1_samples': int(np.sum(players == 1)),
        'player_2_samples': int(np.sum(players == 2)),
        'win_samples': int(np.sum(outcomes == 1)),
        'loss_samples': int(np.sum(outcomes == 0)),
        'move_distribution': {
            'min_row': int(np.min(moves[:, 0])),
            'max_row': int(np.max(moves[:, 0])),
            'min_col': int(np.min(moves[:, 1])),
            'max_col': int(np.max(moves[:, 1])),
        },
        'memory_size_mb': (
            boards.nbytes + moves.nbytes + 
            players.nbytes + outcomes.nbytes
        ) / 1024 / 1024
    }
    
    return stats


def print_dataset_info(
    boards: np.ndarray,
    moves: np.ndarray,
    players: np.ndarray,
    outcomes: np.ndarray,
    name: str = "Dataset"
):
    """
    Print detailed information about a dataset.
    
    Args:
        boards: Board states
        moves: Move positions
        players: Players
        outcomes: Game outcomes
        name: Name for the dataset
    """
    stats = get_dataset_stats(boards, moves, players, outcomes)
    
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Total samples:     {stats['total_samples']:,}")
    print(f"Board shape:       {stats['board_shape']}")
    print(f"Unique boards:     {stats['unique_boards']:,}")
    print(f"Memory size:       {stats['memory_size_mb']:.2f} MB")
    print(f"\nPlayer distribution:")
    print(f"  Player 1:        {stats['player_1_samples']:,} ({stats['player_1_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  Player 2:        {stats['player_2_samples']:,} ({stats['player_2_samples']/stats['total_samples']*100:.1f}%)")
    print(f"\nOutcome distribution:")
    print(f"  Wins:            {stats['win_samples']:,} ({stats['win_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  Losses/Draws:    {stats['loss_samples']:,} ({stats['loss_samples']/stats['total_samples']*100:.1f}%)")
    print(f"{'='*60}\n")
