import pytest
from game.board import Board
from agents.random_agent import RandomAgent

class TestRandomAgent:
    """Test cases for RandomAgent."""
    
    def test_initialization(self):
        """Test agent initializes correctly."""
        agent = RandomAgent(1)
        assert agent.player == 1
        assert agent.get_name() == "Random_P1"
    
    def test_choose_move_returns_legal_move(self):
        """Test that agent only chooses legal moves."""
        agent = RandomAgent(1)
        board = Board(15)
        
        # Make some moves to occupy positions
        board.make_move(7, 7, 1)
        board.make_move(7, 8, 2)
        
        # Agent should choose from legal moves only
        for _ in range(10):  # Try multiple times
            row, col = agent.choose_move(board)
            assert board.is_valid_move(row, col)
            assert (row, col) in board.get_legal_moves()
    
    def test_choose_move_on_empty_board(self):
        """Test agent can choose on empty board."""
        agent = RandomAgent(1)
        board = Board(15)
        
        row, col = agent.choose_move(board)
        assert 0 <= row < 15
        assert 0 <= col < 15
        assert board.is_valid_move(row, col)
    
    def test_choose_move_on_nearly_full_board(self):
        """Test agent can choose when few moves remain."""
        agent = RandomAgent(1)
        board = Board(5)  # Small board
        
        # Fill most of the board
        player = 1
        for row in range(5):
            for col in range(5):
                if not (row == 2 and col == 2):  # Leave one spot open
                    board.make_move(row, col, player)
                    player = 3 - player
        
        # Should choose the only remaining move
        chosen_row, chosen_col = agent.choose_move(board)
        assert (chosen_row, chosen_col) == (2, 2)
    
    def test_different_players(self):
        """Test agents with different player numbers."""
        agent1 = RandomAgent(1)
        agent2 = RandomAgent(2)
        
        assert agent1.player == 1
        assert agent2.player == 2
        assert agent1.get_name() == "Random_P1"
        assert agent2.get_name() == "Random_P2"


class TestGameWithAgents:
    """Test game mechanics with agents."""
    
    def test_complete_game(self):
        """Test that a complete game can be played."""
        from game.rules import Game
        
        agent1 = RandomAgent(1)
        agent2 = RandomAgent(2)
        game = Game(agent1, agent2)
        
        winner = game.play(verbose=False)
        
        # Winner should be 0, 1, or 2
        assert winner in [0, 1, 2]
        
        # Game should have made moves
        assert game.move_count > 0
        
        # Game should end (not infinite loop)
        assert game.move_count <= 225
    
    def test_multiple_games(self):
        """Test multiple games can be played."""
        from game.rules import Game
        
        for _ in range(5):
            agent1 = RandomAgent(1)
            agent2 = RandomAgent(2)
            game = Game(agent1, agent2)
            winner = game.play(verbose=False)
            
            assert winner in [0, 1, 2]
            assert game.move_count > 0