import random
from typing import Tuple
from agents.base_agent import Agent

class RandomAgent(Agent):
    def __init__(self, player: int):
        super().__init__(player)
    
    def choose_move(self, board) -> Tuple[int, int]:
        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        return random.choice(legal_moves)
    
    def get_name(self) -> str:
        return f"Random_P{self.player}"