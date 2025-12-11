#!/usr/bin/env python3
"""
Advanced testing for Minimax agent:
- Verify rule compliance
- Test specific scenarios  
- Performance benchmarking
"""

import time
from game.board import Board
from game.rules import Game
from agents.random_agent import RandomAgent
from agents.minimax import MinimaxAgent


def test_rule_compliance():
    """Test that Minimax never makes illegal moves."""
    print("=" * 60)
    print("TEST: Rule Compliance")
    print("=" * 60)
    
    num_games = 50
    illegal_moves = 0
    
    for i in range(num_games):
        minimax = MinimaxAgent(player_id=1, depth=2)
        random_agent = RandomAgent(player=2)
        game = Game(minimax, random_agent)
        
        # Custom play to check each move
        board = game.board
        current_player = 1
        move_count = 0
        
        while move_count < 225:
            agent = minimax if current_player == 1 else random_agent
            
            try:
                row, col = agent.choose_move(board)
                
                # Verify move is legal
                if not board.is_valid_move(row, col):
                    print(f"  ILLEGAL MOVE by {agent.get_name()} at ({row}, {col})")
                    illegal_moves += 1
                    break
                
                # Make move
                board.make_move(row, col, current_player)
                move_count += 1
                
                # Check win
                if board.check_winner(row, col, current_player):
                    break
                    
                if board.is_full():
                    break
                    
            except Exception as e:
                print(f"  ERROR in game {i+1}: {e}")
                illegal_moves += 1
                break
            
            current_player = 3 - current_player
        
        if (i + 1) % 10 == 0:
            print(f"  Tested {i+1}/{num_games} games...")
    
    print(f"\nResult: {illegal_moves} illegal moves in {num_games} games")
    if illegal_moves == 0:
        print("[PASS] All moves are legal")
    else:
        print("[FAIL] Found illegal moves")
    print()
    return illegal_moves == 0


def test_critical_defense():
    """Test if Minimax can defend against immediate threats."""
    print("=" * 60)
    print("TEST: Critical Defense (Block 4-in-a-row)")
    print("=" * 60)
    
    minimax = MinimaxAgent(player_id=2, depth=2)
    board = Board(15)
    
    # Create a scenario: Player 1 has 4 in a row horizontally
    # .XXXX. at row 7, cols 1-4
    # Minimax (player 2) should block at (7, 0) or (7, 5)
    
    for col in range(1, 5):
        board.make_move(7, col, 1)
    
    print("Board setup:")
    print("  Player 1 has 4 in a row at (7, 1-4)")
    print("  Minimax should block at (7, 0) or (7, 5)")
    
    # Minimax chooses
    row, col = minimax.choose_move(board)
    
    print(f"\nMinimax chose: ({row}, {col})")
    
    # Check if it blocks
    if row == 7 and (col == 0 or col == 5):
        print("[PASS] Correctly blocked the threat")
        return True
    else:
        print("[FAIL] Did not block the threat")
        return False


def test_winning_move():
    """Test if Minimax takes winning move when available."""
    print("\n" + "=" * 60)
    print("TEST: Taking Winning Move")
    print("=" * 60)
    
    minimax = MinimaxAgent(player_id=1, depth=2)
    board = Board(15)
    
    # Create scenario: Minimax (player 1) has 4 in a row
    # XXXX. at row 7, cols 0-3
    # Minimax should complete at (7, 4)
    
    for col in range(4):
        board.make_move(7, col, 1)
    
    print("Board setup:")
    print("  Minimax has 4 in a row at (7, 0-3)")
    print("  Should win by placing at (7, 4)")
    
    row, col = minimax.choose_move(board)
    
    print(f"\nMinimax chose: ({row}, {col})")
    
    if row == 7 and col == 4:
        print("[PASS] Correctly took winning move")
        return True
    else:
        print("[FAIL] Did not take winning move")
        return False


def benchmark_performance():
    """Measure average thinking time per move."""
    print("\n" + "=" * 60)
    print("TEST: Performance Benchmark")
    print("=" * 60)
    
    minimax = MinimaxAgent(player_id=1, depth=2)
    random_agent = RandomAgent(player=2)
    
    thinking_times = []
    num_games = 10
    
    print(f"Running {num_games} games to measure thinking time...")
    
    for i in range(num_games):
        game = Game(minimax, random_agent)
        board = game.board
        current_player = 1
        
        while True:
            if current_player == 1:  # Minimax's turn
                start = time.time()
                row, col = minimax.choose_move(board)
                end = time.time()
                thinking_times.append(end - start)
                
                board.make_move(row, col, 1)
                if board.check_winner(row, col, 1) or board.is_full():
                    break
            else:  # Random's turn
                row, col = random_agent.choose_move(board)
                board.make_move(row, col, 2)
                if board.check_winner(row, col, 2) or board.is_full():
                    break
            
            current_player = 3 - current_player
    
    avg_time = sum(thinking_times) / len(thinking_times)
    max_time = max(thinking_times)
    min_time = min(thinking_times)
    
    print(f"\nTotal moves analyzed: {len(thinking_times)}")
    print(f"Average thinking time: {avg_time*1000:.2f} ms")
    print(f"Min thinking time: {min_time*1000:.2f} ms")
    print(f"Max thinking time: {max_time*1000:.2f} ms")
    
    if avg_time < 1.0:  # Less than 1 second average
        print("[PASS] Performance is acceptable")
        return True
    else:
        print("[WARNING] Thinking time might be too slow")
        return False


def test_different_depths():
    """Test win rates at different depths."""
    print("\n" + "=" * 60)
    print("TEST: Win Rate at Different Depths")
    print("=" * 60)
    
    depths = [1, 2, 3]
    num_games = 20
    
    for depth in depths:
        print(f"\nDepth = {depth}:")
        minimax = MinimaxAgent(player_id=1, depth=depth)
        random_agent = RandomAgent(player=2)
        
        wins = 0
        start_time = time.time()
        
        for i in range(num_games):
            game = Game(minimax, random_agent)
            winner = game.play(verbose=False)
            if winner == 1:
                wins += 1
        
        end_time = time.time()
        win_rate = wins / num_games * 100
        total_time = end_time - start_time
        
        print(f"  Win rate: {win_rate:.1f}% ({wins}/{num_games})")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time/game: {total_time/num_games:.2f}s")


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("ADVANCED MINIMAX TESTING SUITE")
    print("=" * 80)
    print()
    
    # Run all tests
    results = []
    
    results.append(("Rule Compliance", test_rule_compliance()))
    results.append(("Critical Defense", test_critical_defense()))
    results.append(("Winning Move", test_winning_move()))
    results.append(("Performance", benchmark_performance()))
    
    test_different_depths()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:<25} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 80)
