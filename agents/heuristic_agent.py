from typing import Tuple, List
import random
from agents.base_agent import Agent
from game.patterns import PatternDetector


class HeuristicAgent(Agent):
    def __init__(self, player: int, search_depth: int = 1):
        super().__init__(player)
        self.search_depth = search_depth
        self.detector = PatternDetector()
    
    def choose_move(self, board) -> Tuple[int, int]:
        # Check for immediate winning moves or blocks
        my_threats = self.detector.get_threats(board.grid, self.player)
        opponent_threats = self.detector.get_threats(board.grid, 3 - self.player)
        
        if my_threats:
            return my_threats[0]
        
        if opponent_threats:
            return opponent_threats[0]
        
        # Evaluate all legal moves
        legal_moves = board.get_adjacent_moves(distance=2)
        
        if not legal_moves:
            legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # If only one move, take it
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Evaluate each move
        best_move = None
        best_score = float('-inf')
        
        for row, col in legal_moves:
            # Try the move
            board.make_move(row, col, self.player)
            
            # Evaluate the position after this move
            score = self.detector.evaluate_position(board.grid, self.player)
            
            # Add some randomness to avoid predictability
            score += random.uniform(-10, 10)
            
            board.undo_move()
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move if best_move else legal_moves[0]
    
    def get_name(self) -> str:
        return f"Heuristic_P{self.player}"
