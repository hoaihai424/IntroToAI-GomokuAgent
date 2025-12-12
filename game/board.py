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
    
    def undo_move(self) -> Optional[Tuple[int, int, int]]:
        if not self.move_history:
            return None
        
        row, col, player = self.move_history.pop()
        self.grid[row][col] = 0
        return (row, col, player)
    
    def is_valid_move(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == 0
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == 0:
                    moves.append((row, col))
        return moves
    
    def get_adjacent_moves(self, distance: int = 2) -> List[Tuple[int, int]]:
        # Return the center of the board if it's empty
        if np.sum(self.grid) == 0: 
            center = self.size // 2
            return [(center, center)]
        
        candidates = set()
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] != 0:
                    for dr in range(-distance, distance + 1):
                        for dc in range(-distance, distance + 1):
                            new_row, new_col = row + dr, col + dc
                            if self.is_valid_move(new_row, new_col):
                                candidates.add((new_row, new_col))
        
        return list(candidates)
    
    def check_winner(self, row: int, col: int, player: int) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
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
    
    def __str__(self) -> str:
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        
        header = "   " + " ".join(f"{i:2d}" for i in range(self.size))
        lines.append(header)
        
        for i, row in enumerate(self.grid):
            row_str = f"{i:2d} " + " ".join(f" {symbols[cell]}" for cell in row)
            lines.append(row_str)
        
        return "\n".join(lines)