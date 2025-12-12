#!/usr/bin/env python3
"""
Simple evaluation script to test Minimax performance.
"""

from game.rules import Game
from agents.random_agent import RandomAgent
from agents.minimax import MinimaxAgent


def simple_test(num_games=50):
    """Run a simple test and show clear results."""
    print(f"\nTesting Minimax (depth=2) vs Random Agent - {num_games} games")
    print("="*60)
    
    # Test 1: Minimax goes first (Player 1)
    print("\nTest 1: Minimax plays first (X)")
    minimax_first_wins = 0
    for i in range(num_games):
        minimax = MinimaxAgent(player_id=1, depth=2)
        random_agent = RandomAgent(player=2)
        game = Game(minimax, random_agent)
        winner = game.play(verbose=False)
        if winner == 1:  # Minimax wins
            minimax_first_wins += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_games}")
    
    print(f"  Minimax wins: {minimax_first_wins}/{num_games} = {minimax_first_wins/num_games*100:.1f}%")
    
    # Test 2: Minimax goes second (Player 2)
    print("\nTest 2: Minimax plays second (O)")
    minimax_second_wins = 0
    for i in range(num_games):
        random_agent = RandomAgent(player=1)
        minimax = MinimaxAgent(player_id=2, depth=2)
        game = Game(random_agent, minimax)
        winner = game.play(verbose=False)
        if winner == 2:  # Minimax wins
            minimax_second_wins += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_games}")
    
    print(f"  Minimax wins: {minimax_second_wins}/{num_games} = {minimax_second_wins/num_games*100:.1f}%")
    
    # Overall results
    total_games = num_games * 2
    total_wins = minimax_first_wins + minimax_second_wins
    win_rate = total_wins / total_games * 100
    
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Total games: {total_games}")
    print(f"Minimax total wins: {total_wins}/{total_games}")
    print(f"Win rate: {win_rate:.2f}%")
    print()
    
    if win_rate >= 90:
        print(f"[SUCCESS] Win rate {win_rate:.2f}% >= 90%")
    else:
        print(f"[FAILED] Win rate {win_rate:.2f}% < 90%")
        print(f"  Gap: {90 - win_rate:.2f}%")
    
    print("="*60)
    return win_rate


if __name__ == "__main__":
    import sys
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    simple_test(num_games)
