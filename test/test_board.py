import pytest
import numpy as np
from game.board import Board


class TestBoard:
    """Test suite for the Board class."""
    
    def test_board_initialization(self):
        """Test that board initializes correctly."""
        board = Board(15)
        assert board.size == 15
        assert board.grid.shape == (15, 15)
        assert np.all(board.grid == 0)
        assert len(board.move_history) == 0
    
    def test_make_move(self):
        """Test making valid moves."""
        board = Board(15)
        
        # Valid move for player 1
        assert board.make_move(7, 7, 1) == True
        assert board.grid[7][7] == 1
        assert len(board.move_history) == 1
        
        # Valid move for player 2
        assert board.make_move(7, 8, 2) == True
        assert board.grid[7][8] == 2
        assert len(board.move_history) == 2
    
    def test_invalid_move(self):
        """Test that invalid moves are rejected."""
        board = Board(15)
        
        # Out of bounds
        assert board.make_move(-1, 0, 1) == False
        assert board.make_move(0, -1, 1) == False
        assert board.make_move(15, 0, 1) == False
        assert board.make_move(0, 15, 1) == False
        
        # Occupied position
        board.make_move(7, 7, 1)
        assert board.make_move(7, 7, 2) == False
    
    def test_undo_move(self):
        """Test undoing moves."""
        board = Board(15)
        
        # Undo on empty board
        assert board.undo_move() is None
        
        # Make moves and undo
        board.make_move(7, 7, 1)
        board.make_move(7, 8, 2)
        
        result = board.undo_move()
        assert result == (7, 8, 2)
        assert board.grid[7][8] == 0
        assert len(board.move_history) == 1
        
        result = board.undo_move()
        assert result == (7, 7, 1)
        assert board.grid[7][7] == 0
        assert len(board.move_history) == 0
    
    def test_is_valid_move(self):
        """Test move validation."""
        board = Board(15)
        
        # Valid empty position
        assert board.is_valid_move(7, 7) == True
        
        # Out of bounds
        assert board.is_valid_move(-1, 7) == False
        assert board.is_valid_move(15, 7) == False
        
        # Occupied position
        board.make_move(7, 7, 1)
        assert board.is_valid_move(7, 7) == False
    
    def test_get_legal_moves(self):
        """Test getting all legal moves."""
        board = Board(5)  # Use smaller board for testing
        
        # Empty board should have all positions legal
        legal_moves = board.get_legal_moves()
        assert len(legal_moves) == 25
        
        # After one move
        board.make_move(2, 2, 1)
        legal_moves = board.get_legal_moves()
        assert len(legal_moves) == 24
        assert (2, 2) not in legal_moves
    
    def test_get_adjacent_moves_empty_board(self):
        """Test adjacent moves on empty board returns center."""
        board = Board(15)
        adjacent = board.get_adjacent_moves()
        assert len(adjacent) == 1
        assert adjacent[0] == (7, 7)
    
    def test_get_adjacent_moves(self):
        """Test getting adjacent moves with distance."""
        board = Board(15)
        board.make_move(7, 7, 1)
        
        # Distance 1
        adjacent = board.get_adjacent_moves(distance=1)
        assert (7, 7) not in adjacent  # Original position should be occupied
        assert (6, 6) in adjacent or (8, 8) in adjacent  # Some adjacent positions
        
        # Distance 2
        adjacent = board.get_adjacent_moves(distance=2)
        assert len(adjacent) > 0
    
    def test_horizontal_win(self):
        """Test horizontal win detection."""
        board = Board(15)
        
        # Create horizontal line
        for col in range(5):
            board.make_move(7, col, 1)
        
        assert board.check_winner(7, 4, 1) == True
        assert board.check_winner(7, 0, 1) == True
        assert board.check_winner(7, 2, 1) == True
    
    def test_vertical_win(self):
        """Test vertical win detection."""
        board = Board(15)
        
        # Create vertical line
        for row in range(5):
            board.make_move(row, 7, 1)
        
        assert board.check_winner(4, 7, 1) == True
        assert board.check_winner(0, 7, 1) == True
    
    def test_diagonal_win(self):
        """Test diagonal win detection (top-left to bottom-right)."""
        board = Board(15)
        
        # Create diagonal line (\)
        for i in range(5):
            board.make_move(i, i, 1)
        
        assert board.check_winner(4, 4, 1) == True
        assert board.check_winner(0, 0, 1) == True
        assert board.check_winner(2, 2, 1) == True
    
    def test_anti_diagonal_win(self):
        """Test anti-diagonal win detection (top-right to bottom-left)."""
        board = Board(15)
        
        # Create anti-diagonal line (/)
        for i in range(5):
            board.make_move(i, 4 - i, 1)
        
        assert board.check_winner(0, 4, 1) == True
        assert board.check_winner(4, 0, 1) == True
        assert board.check_winner(2, 2, 1) == True
    
    def test_no_win_with_four(self):
        """Test that 4 in a row doesn't win."""
        board = Board(15)
        
        # Create line of 4
        for col in range(4):
            board.make_move(7, col, 1)
        
        assert board.check_winner(7, 3, 1) == False
    
    def test_no_win_different_players(self):
        """Test that mixed players don't trigger win."""
        board = Board(15)
        
        # Create mixed line
        board.make_move(7, 0, 1)
        board.make_move(7, 1, 1)
        board.make_move(7, 2, 2)  # Different player
        board.make_move(7, 3, 1)
        board.make_move(7, 4, 1)
        
        assert board.check_winner(7, 4, 1) == False
    
    def test_win_with_more_than_five(self):
        """Test that 6 or more in a row still wins."""
        board = Board(15)
        
        # Create line of 6
        for col in range(6):
            board.make_move(7, col, 1)
        
        assert board.check_winner(7, 5, 1) == True
    
    def test_is_full(self):
        """Test full board detection."""
        board = Board(3)  # Small board for testing
        
        assert board.is_full() == False
        
        # Fill the board
        for row in range(3):
            for col in range(3):
                board.make_move(row, col, (row + col) % 2 + 1)
        
        assert board.is_full() == True
    
    def test_copy(self):
        """Test board copying."""
        board = Board(15)
        board.make_move(7, 7, 1)
        board.make_move(7, 8, 2)
        
        # Make a copy
        board_copy = board.copy()
        
        # Verify copy has same state
        assert np.array_equal(board_copy.grid, board.grid)
        assert board_copy.move_history == board.move_history
        assert board_copy.size == board.size
        
        # Verify independence
        board_copy.make_move(8, 8, 1)
        assert not np.array_equal(board_copy.grid, board.grid)
    
    def test_get_state(self):
        """Test getting board state."""
        board = Board(15)
        board.make_move(7, 7, 1)
        
        state = board.get_state()
        assert isinstance(state, np.ndarray)
        assert state.shape == (15, 15)
        assert state[7][7] == 1
        
        # Verify it's a copy
        state[0][0] = 99
        assert board.grid[0][0] != 99
    
    def test_str_representation(self):
        """Test string representation."""
        board = Board(5)
        board.make_move(2, 2, 1)
        board.make_move(2, 3, 2)
        
        board_str = str(board)
        assert 'X' in board_str
        assert 'O' in board_str
        assert '.' in board_str