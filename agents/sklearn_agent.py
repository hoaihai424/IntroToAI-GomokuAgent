"""
Sklearn-based agent for Gomoku.
Uses trained MLPClassifier to choose moves.
"""

import numpy as np
import joblib
from typing import Tuple
from agents.base_agent import Agent


class SklearnAgent(Agent):
    """Agent that uses trained sklearn MLPClassifier."""
    
    def __init__(self, player: int, model_path: str, temperature: float = 1.0):
        """
        Initialize sklearn agent.
        
        Args:
            player: Player number (1 or 2)
            model_path: Path to trained model file
            temperature: Sampling temperature (0.5-2.0)
        """
        super().__init__(player)
        self.temperature = temperature
        
        # Load model
        try:
            self.model = joblib.load(model_path)
            print(f"✓ Loaded model for player {player} from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Import helper functions
        from training.sklearn_models import board_to_features, class_to_move
        self.board_to_features = board_to_features
        self.class_to_move = class_to_move
    
    def choose_move(self, board) -> Tuple[int, int]:
        """
        Choose move using trained model.
        
        Args:
            board: Current Board instance
            
        Returns:
            (row, col) tuple for chosen move
        """
        # Get legal moves
        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Fallback for single legal move
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Convert board to features
        features = self.board_to_features(board.grid, self.player)
        
        # Get model predictions
        try:
            probs = self.model.predict_proba([features])[0]
        except Exception as e:
            print(f"Warning: Model prediction failed ({e}), using random move")
            import random
            return random.choice(legal_moves)
        
        # Create mask for legal moves
        mask = np.zeros(225, dtype=np.float32)
        for row, col in legal_moves:
            idx = row * 15 + col
            mask[idx] = 1.0
        
        # Apply mask
        probs = probs * mask
        
        # Check if any legal moves have probability
        if probs.sum() == 0 or np.isnan(probs).any():
            # Fallback to random legal move
            import random
            return random.choice(legal_moves)
        
        # Apply temperature
        if self.temperature != 1.0:
            probs = np.power(probs, 1.0 / self.temperature)
        
        # Check again after temperature
        if probs.sum() == 0 or np.isnan(probs).any() or np.isinf(probs).any():
            # Fallback to random legal move
            import random
            return random.choice(legal_moves)
        
        # Renormalize
        probs = probs / probs.sum()
        
        # Final check
        if np.isnan(probs).any() or np.isinf(probs).any():
            # Fallback to random legal move
            import random
            return random.choice(legal_moves)
        
        # Sample move
        class_idx = np.random.choice(225, p=probs)
        row, col = self.class_to_move(class_idx)
        
        # Verify move is legal (should always be true)
        if not board.is_valid_move(row, col):
            print(f"Warning: Model chose illegal move ({row}, {col}), using random")
            import random
            return random.choice(legal_moves)
        
        return (row, col)
    
    def get_name(self) -> str:
        """Return agent name."""
        return f"Sklearn_P{self.player}"
