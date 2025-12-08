import numpy as np
from typing import List, Tuple, Optional

class Board:
    def __init__(self, size: int = 15):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # 0=empty, 1=player1, 2=player2
        self.move_history = []  # row, col, player

    def reset(self):
        self.grid.fill(0)
        self.move_history.clear()
        
    def make_move(self, row: int, col: int, player: int) -> bool:
        # Check if the move is valid
        if not self.is_valid_move(row, col):
            return False
        
        self.grid[row][col] = player
        self.move_history.append((row, col, player))
        return True
    
    def is_valid_move(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == 0
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == 0:
                    moves.append((row, col))
        return moves
    
    def check_winner(self, row: int, col: int, player: int) -> bool:
        # Check all 4 directions: horizontal, vertical, diagonal1 (\), diagonal2 (/)
        directions = [ (0, 1), (1, 0), (1, 1), (1, -1) ]
        
        for dr, dc in directions:
            count = 1 
            
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == player:
                count += 1
                r += dr
                c += dc
            
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def is_full(self) -> bool:
        return np.all(self.grid != 0)
    
    def get_state(self) -> np.ndarray:
        return self.grid.copy()
    
    def __str__(self) -> str:
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        
        # Column numbers
        header = "   " + " ".join(f"{i:2d}" for i in range(self.size))
        lines.append(header)
        
        for i, row in enumerate(self.grid):
            row_str = f"{i:2d} " + " ".join(f" {symbols[cell]}" for cell in row)
            lines.append(row_str)
        
        return "\n".join(lines)