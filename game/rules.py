from typing import Optional, Tuple
from game.board import Board

class Game:
    def __init__(self, agent1, agent2, board_size: int = 15):
        self.board = Board(board_size)
        self.agent1 = agent1
        self.agent2 = agent2
        self.agents = {1: agent1, 2: agent2}
        self.current_player = 1
        self.winner = None
        self.move_count = 0
        
    def play(self, verbose: bool = False, max_moves: int = 225) -> int:
        """
        Play a complete game.
        
        Args:
            verbose: Whether to print board state and moves
            max_moves: Maximum number of moves before declaring a draw
            
        Returns:
            Winner (1 or 2) or 0 for draw
        """
        if verbose:
            print(f"Game started: {self.agent1.get_name()} (X) vs {self.agent2.get_name()} (O)")
            print(self.board)
            print()
        
        while self.winner is None and self.move_count < max_moves:
            # Get current agent
            current_agent = self.agents[self.current_player]
            
            # Agent chooses move
            try:
                row, col = current_agent.choose_move(self.board)
            except Exception as e:
                if verbose:
                    print(f"Error: {current_agent.get_name()} failed to choose move: {e}")
                # Opponent wins if agent crashes
                self.winner = 3 - self.current_player
                break
            
            # Validate and make move
            if not self.board.is_valid_move(row, col):
                if verbose:
                    print(f"Illegal move by {current_agent.get_name()}: ({row}, {col})")
                # Opponent wins if illegal move
                self.winner = 3 - self.current_player
                break
            
            # Place the stone
            self.board.make_move(row, col, self.current_player)
            self.move_count += 1
            
            if verbose:
                symbol = 'X' if self.current_player == 1 else 'O'
                print(f"Move {self.move_count}: {current_agent.get_name()} ({symbol}) -> ({row}, {col})")
                print(self.board)
                print()
            
            # Check for winner
            if self.board.check_winner(row, col, self.current_player):
                self.winner = self.current_player
                if verbose:
                    print(f"{current_agent.get_name()} wins!")
                break
            
            # Check for draw
            if self.board.is_full():
                self.winner = 0
                if verbose:
                    print("Game is a draw!")
                break
            
            # Switch players
            self.current_player = 3 - self.current_player
        
        # Timeout draw
        if self.winner is None:
            self.winner = 0
            if verbose:
                print(f"Draw by move limit ({max_moves} moves)")
        
        return self.winner
    
    def get_state(self) -> Tuple[Board, int]:
        return self.board, self.current_player
    
    def reset(self):
        self.board = Board(self.board.size)
        self.current_player = 1
        self.winner = None
        self.move_count = 0