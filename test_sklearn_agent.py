#!/usr/bin/env python3
"""
Test script for Sklearn-based Gomoku agent.
Evaluates win rate against different opponents.
"""

import argparse
import numpy as np
from tqdm import tqdm
from game.board import Board
from game.rules import Game
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.sklearn_agent import SklearnAgent


def test_agent_matchup(agent1, agent2, num_games=100, verbose=False):
    """
    Test two agents against each other.
    
    Args:
        agent1: First agent
        agent2: Second agent  
        num_games: Number of games to play
        verbose: Print game details
        
    Returns:
        Dictionary with results
    """
    results = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'game_lengths': []
    }
    
    iterator = tqdm(range(num_games), desc=f"{agent1.get_name()} vs {agent2.get_name()}")
    
    for game_num in iterator:
        game = Game(agent1, agent2, board_size=15)
        
        try:
            winner = game.play(verbose=False)
            game_length = len(game.board.move_history)
            results['game_lengths'].append(game_length)
            
            if winner == 1:
                results['agent1_wins'] += 1
            elif winner == 2:
                results['agent2_wins'] += 1
            else:
                results['draws'] += 1
                
            if verbose and (game_num + 1) % 10 == 0:
                print(f"Game {game_num + 1}: Winner=P{winner}, Length={game_length}")
        
        except Exception as e:
            print(f"Error in game {game_num + 1}: {e}")
            results['draws'] += 1
    
    return results


def print_results(results, agent1_name, agent2_name, num_games):
    """Print test results in formatted table."""
    print("\n" + "="*70)
    print(f"RESULTS: {agent1_name} vs {agent2_name}")
    print("="*70)
    
    agent1_wins = results['agent1_wins']
    agent2_wins = results['agent2_wins']
    draws = results['draws']
    
    print(f"\nTotal Games:  {num_games}")
    print(f"  {agent1_name} wins:  {agent1_wins} ({100*agent1_wins/num_games:.1f}%)")
    print(f"  {agent2_name} wins:  {agent2_wins} ({100*agent2_wins/num_games:.1f}%)")
    print(f"  Draws:        {draws} ({100*draws/num_games:.1f}%)")
    
    if results['game_lengths']:
        avg_length = np.mean(results['game_lengths'])
        min_length = np.min(results['game_lengths'])
        max_length = np.max(results['game_lengths'])
        
        print(f"\nGame Length Statistics:")
        print(f"  Average: {avg_length:.1f} moves")
        print(f"  Min:     {min_length} moves")
        print(f"  Max:     {max_length} moves")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test Sklearn Gomoku Agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained sklearn model')
    parser.add_argument('--opponent', type=str, default='random',
                       choices=['random', 'heuristic', 'self'],
                       help='Opponent type')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games to play')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature for sklearn agent')
    parser.add_argument('--both-sides', action='store_true',
                       help='Test with sklearn as both player 1 and 2')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed game information')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SKLEARN AGENT TESTING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model:        {args.model}")
    print(f"  Opponent:     {args.opponent}")
    print(f"  Games:        {args.num_games}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Both sides:   {args.both_sides}")
    print("="*70)
    
    # Test 1: Sklearn as Player 1
    print(f"\n📊 Test 1: Sklearn (P1) vs {args.opponent.capitalize()} (P2)")
    print("-"*70)
    
    sklearn_agent = SklearnAgent(1, args.model, temperature=args.temperature)
    
    if args.opponent == 'random':
        opponent = RandomAgent(2)
    elif args.opponent == 'heuristic':
        opponent = HeuristicAgent(2)
    elif args.opponent == 'self':
        opponent = SklearnAgent(2, args.model, temperature=args.temperature)
    
    results1 = test_agent_matchup(sklearn_agent, opponent, args.num_games, args.verbose)
    print_results(results1, "Sklearn Agent", opponent.get_name(), args.num_games)
    
    # Test 2: Sklearn as Player 2 (if requested)
    if args.both_sides:
        print(f"\n📊 Test 2: {args.opponent.capitalize()} (P1) vs Sklearn (P2)")
        if args.opponent == 'random':
            opponent = RandomAgent(1)
        elif args.opponent == 'heuristic':
            opponent = HeuristicAgent(1)
        elif args.opponent == 'self':
            opponent = SklearnAgent(1, args.model, temperature=args.temperature)
        
        sklearn_agent = SklearnAgent(2, args.model, temperature=args.temperature)
        
        results2 = test_agent_matchup(opponent, sklearn_agent, args.num_games, args.verbose)
        print_results(results2, opponent.get_name(), "Sklearn Agent", args.num_games)
        
        # Combined statistics
        print("\n" + "="*70)
        print("COMBINED RESULTS (Both as P1 and P2)")
        print("="*70)
        
        total_games = args.num_games * 2
        sklearn_wins = results1['agent1_wins'] + results2['agent2_wins']
        opponent_wins = results1['agent2_wins'] + results2['agent1_wins']
        draws = results1['draws'] + results2['draws']
        
        print(f"\nTotal Games:  {total_games}")
        print(f"  Sklearn wins:   {sklearn_wins} ({100*sklearn_wins/total_games:.1f}%)")
        print(f"  Opponent wins:  {opponent_wins} ({100*opponent_wins/total_games:.1f}%)")
        print(f"  Draws:          {draws} ({100*draws/total_games:.1f}%)")
        print("="*70 + "\n")
    
    # Performance assessment
    win_rate = results1['agent1_wins'] / args.num_games
    print("\n🎯 Performance Assessment:")
    if win_rate >= 0.80:
        print("   🌟 EXCELLENT - Model performing very well!")
    elif win_rate >= 0.70:
        print("   ✅ GOOD - Model meeting expectations!")
    elif win_rate >= 0.60:
        print("   📈 ACCEPTABLE - Model shows promise, room for improvement")
    elif win_rate >= 0.50:
        print("   ⚠️  NEEDS IMPROVEMENT - Consider more training or better data")
    else:
        print("   ❌ POOR - Model needs significant improvement")
    
    print(f"\nNext steps:")
    print(f"  1. If performance good: Test in GUI (python play_gomoku.py)")
    print(f"  2. If performance poor: Generate more/better training data")
    print(f"  3. Try different temperature values (0.5-2.0)")
    print()


if __name__ == "__main__":
    main()
