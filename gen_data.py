#!/usr/bin/env python3
"""
Generate training data for ML agent by playing games between agents.

This script:
1. Plays games between different agents (Random, Heuristic)
2. Extracts board states and moves from each game
3. Applies data augmentation (8-fold symmetry)
4. Saves the dataset for training

Usage:
    python gen_data.py --games 1000 --agent random
    python gen_data.py --games 500 --agent heuristic --output data/heuristic_games.npz
    python gen_data.py --games 2000 --mixed --augment
"""

import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import os

from game.board import Board
from game.rules import Game
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.sklearn_agent import SklearnAgent
from training.data_augmentation import apply_symmetries
from training.dataset_utils import save_dataset, print_dataset_info


class GameDataGenerator:
    def __init__(self, board_size: int = 15, sklearn_model_path: str = None):
        self.board_size = board_size
        self.sklearn_model_path = sklearn_model_path
        self.samples = []
    
    def play_and_record_game(
        self, 
        agent1, 
        agent2, 
        record_all_moves: bool = True
    ) -> List[Tuple[np.ndarray, Tuple[int, int], int, int]]:
        """
        Play a game and record all positions.
        
        Args:
            agent1: First agent
            agent2: Second agent
            record_all_moves: Whether to record all moves or only winning player's moves
            
        Returns:
            List of (board_state, move, player, outcome) tuples
        """
        game = Game(agent1, agent2, self.board_size)
        positions = []
        
        # Play the game and record each move
        move_count = 0
        max_moves = self.board_size * self.board_size
        
        while game.winner is None and move_count < max_moves:
            current_agent = game.agents[game.current_player]
            
            # Get the board state before the move
            board_before = game.board.grid.copy()
            
            # Agent chooses move
            try:
                row, col = current_agent.choose_move(game.board)
            except:
                break
            
            # Validate move
            if not game.board.is_valid_move(row, col):
                break
            
            # Record the position
            positions.append({
                'board': board_before,
                'move': (row, col),
                'player': game.current_player,
            })
            
            # Make the move
            game.board.make_move(row, col, game.current_player)
            move_count += 1
            
            # Check for winner
            if game.board.check_winner(row, col, game.current_player):
                game.winner = game.current_player
                break
            
            # Check for draw
            if game.board.is_full():
                game.winner = 0
                break
            
            # Switch players
            game.current_player = 3 - game.current_player
        
        # Assign outcomes to positions
        samples = []
        for pos in positions:
            player = pos['player']
            
            # Outcome: 1 if this player won, 0 otherwise
            if game.winner == player:
                outcome = 1
            else:
                outcome = 0
            
            # Only record winning player's moves if specified
            if record_all_moves or outcome == 1:
                samples.append((
                    pos['board'],
                    pos['move'],
                    pos['player'],
                    outcome
                ))
        
        return samples
    
    def generate_dataset(
        self,
        num_games: int,
        agent_type: str = 'random',
        apply_augmentation: bool = True,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate dataset by playing multiple games.
        
        Args:
            num_games: Number of games to play
            agent_type: Type of agent ('random', 'heuristic', 'greedy', 'mixed')
            apply_augmentation: Whether to apply 8-fold augmentation
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (boards, moves, players, outcomes)
        """
        all_boards = []
        all_moves = []
        all_players = []
        all_outcomes = []
        
        iterator = tqdm(range(num_games), desc="Generating games") if show_progress else range(num_games)
        
        for _ in iterator:
            # Create agents based on type
            if agent_type == 'random':
                agent1 = RandomAgent(1)
                agent2 = RandomAgent(2)
            elif agent_type == 'heuristic':
                agent1 = HeuristicAgent(1)
                agent2 = HeuristicAgent(2)
            elif agent_type == 'sklearn':
                if not self.sklearn_model_path:
                    raise ValueError("sklearn_model_path must be provided for sklearn agent")
                agent1 = SklearnAgent(1, self.sklearn_model_path, temperature=1.0)
                agent2 = SklearnAgent(2, self.sklearn_model_path, temperature=1.0)
            elif agent_type == 'sklearn_vs_heuristic':
                if not self.sklearn_model_path:
                    raise ValueError("sklearn_model_path must be provided for sklearn agent")
                agent1 = SklearnAgent(1, self.sklearn_model_path, temperature=0.8)
                agent2 = HeuristicAgent(2)
            elif agent_type == 'heuristic_vs_sklearn':
                if not self.sklearn_model_path:
                    raise ValueError("sklearn_model_path must be provided for sklearn agent")
                agent1 = HeuristicAgent(1)
                agent2 = SklearnAgent(2, self.sklearn_model_path, temperature=0.8)
            elif agent_type == 'mixed':
                # Mix of different agents
                import random
                agents = [RandomAgent, HeuristicAgent]
                Agent1 = random.choice(agents)
                Agent2 = random.choice(agents)
                agent1 = Agent1(1)
                agent2 = Agent2(2)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Play game and get samples
            samples = self.play_and_record_game(agent1, agent2)
            
            # Process each sample
            for board, move, player, outcome in samples:
                if apply_augmentation:
                    # Apply all 8 symmetries
                    augmented = apply_symmetries(board, move, player)
                    for aug_board, aug_move, aug_player in augmented:
                        all_boards.append(aug_board)
                        all_moves.append(aug_move)
                        all_players.append(aug_player)
                        all_outcomes.append(outcome)
                else:
                    # No augmentation
                    all_boards.append(board)
                    all_moves.append(move)
                    all_players.append(player)
                    all_outcomes.append(outcome)
        
        # Convert to numpy arrays
        boards = np.array(all_boards, dtype=np.int8)
        moves = np.array(all_moves, dtype=np.int16)
        players = np.array(all_players, dtype=np.int8)
        outcomes = np.array(all_outcomes, dtype=np.int8)
        
        return boards, moves, players, outcomes

if __name__ == "__main__":
    """Main function for data generation."""
    parser = argparse.ArgumentParser(description='Generate training data for Gomoku ML agent')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to play')
    parser.add_argument('--agent', type=str, default='random', 
                        choices=['random', 'heuristic', 'mixed', 'sklearn', 'sklearn_vs_heuristic', 'heuristic_vs_sklearn'],
                        help='Type of agent to use for self-play')
    parser.add_argument('--output', type=str, default='data/training_data.npz',
                        help='Output file path')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--board-size', type=int, default=15,
                        help='Board size (default: 15)')
    parser.add_argument('--sklearn-model', type=str, default=None,
                        help='Path to sklearn model (required for sklearn agent types)')
    parser.add_argument('--combined', action='store_true',
                        help='Generate combined dataset: sklearn vs heuristic + sklearn vs sklearn')
    args = parser.parse_args()

    # Validate sklearn model path if needed
    if args.agent in ['sklearn', 'sklearn_vs_heuristic', 'heuristic_vs_sklearn'] or args.combined:
        if not args.sklearn_model:
            parser.error("--sklearn-model is required when using sklearn agent types or --combined")

    print("\n" + "="*70)
    print("GOMOKU TRAINING DATA GENERATION")
    print("="*70)
    
    if args.combined:
        # Generate combined dataset: sklearn vs heuristic + sklearn vs sklearn
        print(f"Configuration: COMBINED DATASET")
        print(f"  Games sklearn vs heuristic:  {args.games}")
        print(f"  Games sklearn vs sklearn:    {args.games}")
        print(f"  Total games:                 {args.games * 2}")
        print(f"  Sklearn model:               {args.sklearn_model}")
        print(f"  Board size:                  {args.board_size}x{args.board_size}")
        print(f"  Augmentation:                {'Disabled' if args.no_augment else 'Enabled (8x)'}")
        print(f"  Output file:                 {args.output}")
        print("="*70 + "\n")

        # Generate sklearn vs heuristic data
        print("📊 Phase 1: Generating sklearn vs heuristic games")
        generator1 = GameDataGenerator(board_size=args.board_size, sklearn_model_path=args.sklearn_model)
        boards1, moves1, players1, outcomes1 = generator1.generate_dataset(
            num_games=args.games,
            agent_type='sklearn_vs_heuristic',
            apply_augmentation=not args.no_augment,
            show_progress=True
        )
        
        # Generate heuristic vs sklearn data (swap sides)
        print("\n📊 Phase 2: Generating heuristic vs sklearn games")
        generator2 = GameDataGenerator(board_size=args.board_size, sklearn_model_path=args.sklearn_model)
        boards2, moves2, players2, outcomes2 = generator2.generate_dataset(
            num_games=args.games,
            agent_type='heuristic_vs_sklearn',
            apply_augmentation=not args.no_augment,
            show_progress=True
        )
        
        # Generate sklearn vs sklearn data
        print("\n📊 Phase 3: Generating sklearn vs sklearn games")
        generator3 = GameDataGenerator(board_size=args.board_size, sklearn_model_path=args.sklearn_model)
        boards3, moves3, players3, outcomes3 = generator3.generate_dataset(
            num_games=args.games,
            agent_type='sklearn',
            apply_augmentation=not args.no_augment,
            show_progress=True
        )
        
        # Combine all datasets
        print("\n📊 Combining datasets...")
        boards = np.concatenate([boards1, boards2, boards3])
        moves = np.concatenate([moves1, moves2, moves3])
        players = np.concatenate([players1, players2, players3])
        outcomes = np.concatenate([outcomes1, outcomes2, outcomes3])
        
        # Create metadata
        metadata = {
            'num_games_sklearn_vs_heuristic': args.games * 2,
            'num_games_sklearn_vs_sklearn': args.games,
            'total_games': args.games * 3,
            'sklearn_model': args.sklearn_model,
            'board_size': args.board_size,
            'augmentation': not args.no_augment,
            'total_samples': len(boards),
        }
        
    else:
        # Standard single-type generation
        print(f"Configuration:")
        print(f"  Games to play:    {args.games}")
        print(f"  Agent type:       {args.agent}")
        if args.sklearn_model:
            print(f"  Sklearn model:    {args.sklearn_model}")
        print(f"  Board size:       {args.board_size}x{args.board_size}")
        print(f"  Augmentation:     {'Disabled' if args.no_augment else 'Enabled (8x)'}")
        print(f"  Output file:      {args.output}")
        print("="*70 + "\n")

        # Generate data
        generator = GameDataGenerator(board_size=args.board_size, sklearn_model_path=args.sklearn_model)

        boards, moves, players, outcomes = generator.generate_dataset(
            num_games=args.games,
            agent_type=args.agent,
            apply_augmentation=not args.no_augment,
            show_progress=True
        )

        # Create metadata
        metadata = {
            'num_games': args.games,
            'agent_type': args.agent,
            'board_size': args.board_size,
            'augmentation': not args.no_augment,
            'total_samples': len(boards),
        }
        if args.sklearn_model:
            metadata['sklearn_model'] = args.sklearn_model

    # Print statistics
    print_dataset_info(boards, moves, players, outcomes, "Generated Dataset")

    # Save dataset
    save_dataset(boards, moves, players, outcomes, args.output, metadata)

    print("\n✓ Data generation complete!")
    print(f"  Dataset saved to: {args.output}")
    print(f"  Total samples: {len(boards):,}")
    if not args.no_augment:
        total_games = args.games * 3 if args.combined else args.games
        print(f"  (Original games: {total_games}, Augmentation factor: 8x)")
    print()

