#!/usr/bin/env python3
"""
Evaluation script for testing agent performance.
This script runs multiple games between agents and calculates win rates.
"""

import sys
import time
from typing import Dict, List, Tuple
from game.board import Board
from game.rules import Game
from agents.random_agent import RandomAgent
from agents.minimax import MinimaxAgent


class AgentEvaluator:
    """Evaluates agent performance through multiple games."""
    
    def __init__(self):
        self.results = []
    
    def evaluate(self, agent1, agent2, num_games: int = 100, verbose: bool = True) -> Dict:
        """
        Run multiple games between two agents and calculate statistics.
        
        Args:
            agent1: First agent (will play as Player 1 = X)
            agent2: Second agent (will play as Player 2 = O)
            num_games: Number of games to play
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print("=" * 80)
            print(f"EVALUATION: {agent1.get_name()} vs {agent2.get_name()}")
            print("=" * 80)
            print(f"Running {num_games} games...")
            print()
        
        wins = {1: 0, 2: 0, 0: 0}  # Player 1, Player 2, Draw
        move_counts = []
        game_times = []
        
        start_time = time.time()
        
        for i in range(num_games):
            if verbose and (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_games} games completed...", end='\r')
            
            game = Game(agent1, agent2)
            game_start = time.time()
            winner = game.play(verbose=False)
            game_end = time.time()
            
            wins[winner] += 1
            move_counts.append(game.move_count)
            game_times.append(game_end - game_start)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        results = {
            'agent1_name': agent1.get_name(),
            'agent2_name': agent2.get_name(),
            'total_games': num_games,
            'agent1_wins': wins[1],
            'agent2_wins': wins[2],
            'draws': wins[0],
            'agent1_win_rate': wins[1] / num_games * 100,
            'agent2_win_rate': wins[2] / num_games * 100,
            'draw_rate': wins[0] / num_games * 100,
            'avg_moves': sum(move_counts) / len(move_counts),
            'min_moves': min(move_counts),
            'max_moves': max(move_counts),
            'avg_game_time': sum(game_times) / len(game_times),
            'total_time': total_time
        }
        
        if verbose:
            self.print_results(results)
        
        self.results.append(results)
        return results
    
    def print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Agent 1: {results['agent1_name']}")
        print(f"Agent 2: {results['agent2_name']}")
        print(f"Total games: {results['total_games']}")
        print()
        print(f"Agent 1 (X) wins: {results['agent1_wins']:3d} ({results['agent1_win_rate']:6.2f}%)")
        print(f"Agent 2 (O) wins: {results['agent2_wins']:3d} ({results['agent2_win_rate']:6.2f}%)")
        print(f"Draws:            {results['draws']:3d} ({results['draw_rate']:6.2f}%)")
        print()
        print(f"Average moves per game: {results['avg_moves']:.1f}")
        print(f"Move range: {results['min_moves']} - {results['max_moves']}")
        print(f"Average time per game: {results['avg_game_time']:.3f}s")
        print(f"Total evaluation time: {results['total_time']:.2f}s")
        print()
        
        # Highlight if requirement is met (90% for Minimax vs Random)
        if 'Minimax' in results['agent1_name'] and 'Random' in results['agent2_name']:
            print("-" * 80)
            if results['agent1_win_rate'] >= 90:
                print(f"[PASS] REQUIREMENT MET: Minimax win rate ({results['agent1_win_rate']:.2f}%) >= 90%")
            else:
                print(f"[FAIL] REQUIREMENT NOT MET: Minimax win rate ({results['agent1_win_rate']:.2f}%) < 90%")
                print(f"   Need to improve by: {90 - results['agent1_win_rate']:.2f}%")
            print("-" * 80)
        print()


def test_minimax_vs_random(depth: int = 2, num_games: int = 100):
    """Test Minimax agent against Random agent."""
    print("\n" + "=" * 80)
    print(f"Testing Minimax (depth={depth}) vs Random Agent")
    print("=" * 80 + "\n")
    
    evaluator = AgentEvaluator()
    
    # Test 1: Minimax as Player 1 (X) - goes first
    print("TEST 1: Minimax plays first (X)")
    minimax = MinimaxAgent(player_id=1, depth=depth)
    random_agent = RandomAgent(player=2)
    results1 = evaluator.evaluate(minimax, random_agent, num_games=num_games)
    
    # Test 2: Minimax as Player 2 (O) - goes second
    print("TEST 2: Minimax plays second (O)")
    random_agent = RandomAgent(player=1)
    minimax = MinimaxAgent(player_id=2, depth=depth)
    results2 = evaluator.evaluate(random_agent, minimax, num_games=num_games)
    
    # Combined results
    print("=" * 80)
    print("COMBINED RESULTS (Both positions)")
    print("=" * 80)
    total_games = results1['total_games'] + results2['total_games']
    minimax_wins = results1['agent1_wins'] + results2['agent2_wins']
    random_wins = results1['agent2_wins'] + results2['agent1_wins']
    draws = results1['draws'] + results2['draws']
    
    minimax_win_rate = minimax_wins / total_games * 100
    
    print(f"Total games: {total_games}")
    print(f"Minimax wins: {minimax_wins} ({minimax_win_rate:.2f}%)")
    print(f"Random wins:  {random_wins} ({random_wins / total_games * 100:.2f}%)")
    print(f"Draws:        {draws} ({draws / total_games * 100:.2f}%)")
    print()
    
    if minimax_win_rate >= 90:
        print(f"[PASS] OVERALL REQUIREMENT MET: {minimax_win_rate:.2f}% >= 90%")
    else:
        print(f"[FAIL] OVERALL REQUIREMENT NOT MET: {minimax_win_rate:.2f}% < 90%")
        print(f"   Gap: {90 - minimax_win_rate:.2f}%")
    print("=" * 80)
    
    return minimax_win_rate


def test_different_depths():
    """Test Minimax with different depth settings."""
    print("\n" + "=" * 80)
    print("Testing Minimax with Different Depths")
    print("=" * 80 + "\n")
    
    depths = [1, 2, 3]
    results = []
    
    for depth in depths:
        print(f"\n{'='*80}")
        print(f"Testing Depth = {depth}")
        print(f"{'='*80}")
        
        minimax = MinimaxAgent(player_id=1, depth=depth)
        random_agent = RandomAgent(player=2)
        
        evaluator = AgentEvaluator()
        start = time.time()
        result = evaluator.evaluate(minimax, random_agent, num_games=50, verbose=True)
        end = time.time()
        
        results.append({
            'depth': depth,
            'win_rate': result['agent1_win_rate'],
            'avg_time': result['avg_game_time'],
            'total_time': end - start
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("DEPTH COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Depth':<10} {'Win Rate':<15} {'Avg Time/Game':<20} {'Total Time':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['depth']:<10} {r['win_rate']:>6.2f}%{'':<8} {r['avg_time']:>8.3f}s{'':<11} {r['total_time']:>8.2f}s")
    print("=" * 80)


def quick_test():
    """Quick test with just 20 games for rapid feedback."""
    print("\n[QUICK TEST] 20 games")
    test_minimax_vs_random(depth=2, num_games=20)


def standard_test():
    """Standard test with 100 games."""
    print("\n[STANDARD TEST] 100 games")
    return test_minimax_vs_random(depth=2, num_games=100)


def comprehensive_test():
    """Comprehensive test with 200 games."""
    print("\n[COMPREHENSIVE TEST] 200 games")
    return test_minimax_vs_random(depth=2, num_games=200)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            quick_test()
        elif command == "standard":
            standard_test()
        elif command == "comprehensive":
            comprehensive_test()
        elif command == "depths":
            test_different_depths()
        elif command.startswith("depth="):
            depth = int(command.split("=")[1])
            num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            test_minimax_vs_random(depth=depth, num_games=num_games)
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  quick          - Quick test (20 games)")
            print("  standard       - Standard test (100 games)")
            print("  comprehensive  - Comprehensive test (200 games)")
            print("  depths         - Test different depths")
            print("  depth=N [games]- Test specific depth with N games")
    else:
        # Default: standard test
        standard_test()
