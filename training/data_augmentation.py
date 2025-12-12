"""
Data augmentation for Gomoku training data.

Applies 8-fold symmetry (rotations and reflections) to board states
to increase dataset size and improve model generalization.
"""

import numpy as np
from typing import Tuple, List


def rotate_90(board: np.ndarray, move: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Rotate board and move 90 degrees clockwise.
    
    Args:
        board: Board state (N x N array)
        move: Move position (row, col)
        
    Returns:
        Rotated board and transformed move position
    """
    rotated_board = np.rot90(board, k=-1)  # -1 for clockwise
    size = board.shape[0]
    row, col = move
    new_row = col
    new_col = size - 1 - row
    return rotated_board, (new_row, new_col)


def flip_horizontal(board: np.ndarray, move: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Flip board and move horizontally.
    
    Args:
        board: Board state (N x N array)
        move: Move position (row, col)
        
    Returns:
        Flipped board and transformed move position
    """
    flipped_board = np.fliplr(board)
    size = board.shape[0]
    row, col = move
    new_col = size - 1 - col
    return flipped_board, (row, new_col)


def flip_vertical(board: np.ndarray, move: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Flip board and move vertically.
    
    Args:
        board: Board state (N x N array)
        move: Move position (row, col)
        
    Returns:
        Flipped board and transformed move position
    """
    flipped_board = np.flipud(board)
    size = board.shape[0]
    row, col = move
    new_row = size - 1 - row
    return flipped_board, (new_row, col)


def apply_symmetries(
    board: np.ndarray, 
    move: Tuple[int, int],
    player: int
) -> List[Tuple[np.ndarray, Tuple[int, int], int]]:
    """
    Apply all 8 symmetries to a board state and move.
    
    The 8 symmetries are:
    1. Original
    2. Rotate 90°
    3. Rotate 180°
    4. Rotate 270°
    5. Flip horizontal
    6. Flip vertical
    7. Flip horizontal + rotate 90°
    8. Flip vertical + rotate 90°
    
    Args:
        board: Board state (N x N array)
        move: Move position (row, col)
        player: Player who made the move
        
    Returns:
        List of (augmented_board, augmented_move, player) tuples
    """
    augmented = []
    
    # 1. Original
    augmented.append((board.copy(), move, player))
    
    # 2. Rotate 90°
    board_r90, move_r90 = rotate_90(board, move)
    augmented.append((board_r90, move_r90, player))
    
    # 3. Rotate 180°
    board_r180, move_r180 = rotate_90(board_r90, move_r90)
    augmented.append((board_r180, move_r180, player))
    
    # 4. Rotate 270°
    board_r270, move_r270 = rotate_90(board_r180, move_r180)
    augmented.append((board_r270, move_r270, player))
    
    # 5. Flip horizontal
    board_fh, move_fh = flip_horizontal(board, move)
    augmented.append((board_fh, move_fh, player))
    
    # 6. Flip vertical
    board_fv, move_fv = flip_vertical(board, move)
    augmented.append((board_fv, move_fv, player))
    
    # 7. Flip horizontal + rotate 90°
    board_fh_r90, move_fh_r90 = rotate_90(board_fh, move_fh)
    augmented.append((board_fh_r90, move_fh_r90, player))
    
    # 8. Flip vertical + rotate 90°
    board_fv_r90, move_fv_r90 = rotate_90(board_fv, move_fv)
    augmented.append((board_fv_r90, move_fv_r90, player))
    
    return augmented


def augment_board_state(
    board: np.ndarray,
    move: Tuple[int, int],
    player: int,
    num_augmentations: int = 8
) -> List[Tuple[np.ndarray, Tuple[int, int], int]]:
    """
    Augment a single board state with symmetries.
    
    Args:
        board: Board state
        move: Move position
        player: Current player
        num_augmentations: Number of augmentations (1-8, default: 8 for all)
        
    Returns:
        List of augmented samples
    """
    all_augmented = apply_symmetries(board, move, player)
    return all_augmented[:num_augmentations]


def augment_dataset(
    boards: np.ndarray,
    moves: np.ndarray,
    players: np.ndarray,
    num_augmentations: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment an entire dataset with symmetries.
    
    Args:
        boards: Array of board states (N, H, W)
        moves: Array of moves (N, 2)
        players: Array of players (N,)
        num_augmentations: Number of augmentations per sample
        
    Returns:
        Augmented (boards, moves, players)
    """
    augmented_boards = []
    augmented_moves = []
    augmented_players = []
    
    for i in range(len(boards)):
        board = boards[i]
        move = tuple(moves[i])
        player = players[i]
        
        # Apply augmentations
        aug_samples = augment_board_state(board, move, player, num_augmentations)
        
        for aug_board, aug_move, aug_player in aug_samples:
            augmented_boards.append(aug_board)
            augmented_moves.append(aug_move)
            augmented_players.append(aug_player)
    
    return (
        np.array(augmented_boards),
        np.array(augmented_moves),
        np.array(augmented_players)
    )


def verify_augmentation(board: np.ndarray, move: Tuple[int, int], player: int):
    """
    Verify that augmentations preserve the game state correctly.
    Useful for debugging.
    
    Args:
        board: Original board
        move: Original move
        player: Player
    """
    print("Original board:")
    print(board)
    print(f"Move: {move}, Player: {player}")
    print()
    
    augmented = apply_symmetries(board, move, player)
    
    for i, (aug_board, aug_move, aug_player) in enumerate(augmented):
        print(f"Augmentation {i+1}:")
        print(aug_board)
        print(f"Move: {aug_move}, Player: {aug_player}")
        
        # Verify the move position has the correct player
        row, col = aug_move
        if aug_board[row, col] != player:
            print(f"⚠️  WARNING: Move position doesn't match player!")
        print()
