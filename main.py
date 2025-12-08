#!/usr/bin/env python3
"""
Main entry point for Gomoku agent testing.
"""

from game.board import Board
from game.rules import Game
from agents.random_agent import RandomAgent

def test_board():
    """Test basic board functionality."""
    print("=" * 60)
    print("Testing Board Functionality")
    print("=" * 60)
    
    board = Board(15)
    
    # Make some moves
    print("\nMaking test moves...")
    board.make_move(7, 7, 1)
    board.make_move(7, 8, 2)
    board.make_move(8, 8, 1)
    
    print(board)
    print(f"\nLegal moves: {len(board.get_legal_moves())}")
    print(f"Adjacent moves: {len(board.get_adjacent_moves())}")
    
    # Test win detection
    print("\n" + "=" * 60)
    print("Testing Horizontal Win Detection")
    print("=" * 60)
    test_board = Board(15)
    for col in range(5):
        test_board.make_move(7, col, 1)
    
    print(test_board)
    print(f"Player 1 wins: {test_board.check_winner(7, 4, 1)}")
    

def test_random_vs_random(num_games=5):
    """Test Random agent vs Random agent."""
    print("\n" + "=" * 60)
    print(f"Testing Random vs Random ({num_games} games)")
    print("=" * 60)
    
    wins = {1: 0, 2: 0, 0: 0}  # Track wins for player 1, 2, and draws
    total_moves = []
    
    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        
        agent1 = RandomAgent(1)
        agent2 = RandomAgent(2)
        game = Game(agent1, agent2)
        
        # Play with verbose=True for first game only
        winner = game.play(verbose=(i == 0))
        
        wins[winner] += 1
        total_moves.append(game.move_count)
        
        if not (i == 0):  # Print summary for non-verbose games
            print(f"Winner: {'Player 1' if winner == 1 else 'Player 2' if winner == 2 else 'Draw'} "
                  f"in {game.move_count} moves")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Player 1 (X) wins: {wins[1]} ({wins[1]/num_games*100:.1f}%)")
    print(f"Player 2 (O) wins: {wins[2]} ({wins[2]/num_games*100:.1f}%)")
    print(f"Draws: {wins[0]} ({wins[0]/num_games*100:.1f}%)")
    print(f"Average moves per game: {sum(total_moves)/len(total_moves):.1f}")
    print(f"Min moves: {min(total_moves)}, Max moves: {max(total_moves)}")


def play_single_game_verbose():
    """Play a single game with full output."""
    print("\n" + "=" * 60)
    print("Playing Single Verbose Game")
    print("=" * 60)
    
    agent1 = RandomAgent(1)
    agent2 = RandomAgent(2)
    game = Game(agent1, agent2)
    
    winner = game.play(verbose=True)
    
    print("\n" + "=" * 60)
    print(f"Final Result: {'Player 1 (X)' if winner == 1 else 'Player 2 (O)' if winner == 2 else 'Draw'} wins!")
    print(f"Total moves: {game.move_count}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test-board":
            test_board()
        elif command == "test-random":
            num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            test_random_vs_random(num_games)
        elif command == "play":
            play_single_game_verbose()
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  test-board       - Test board functionality")
            print("  test-random [N]  - Test N random vs random games (default: 10)")
            print("  play             - Play a single verbose game")
    else:
        # Default: run all tests
        test_board()
        play_single_game_verbose()
        test_random_vs_random(10)