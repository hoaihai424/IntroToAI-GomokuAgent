"""
Dual Match Visualization GUI for Gomoku.
Displays two simultaneous games: Sklearn vs Random and Minimax vs Random.
"""

import pygame
import sys
import os
import threading
import time
from typing import Optional, Tuple, List
from game.board import Board
from game.rules import Game
from agents.random_agent import RandomAgent
from agents.sklearn_agent import SklearnAgent
from agents.minimax import MinimaxAgent

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (205, 170, 125)
DARK_BROWN = (165, 130, 95)
GRID_COLOR = (50, 50, 50)
LAST_MOVE_COLOR = (0, 0, 255)
WIN_HIGHLIGHT = (255, 215, 0)
TEXT_COLOR = (50, 50, 50)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 255)
GRAY = (150, 150, 150)

# Board settings
BOARD_SIZE = 15
CELL_SIZE = 30  # Smaller for dual view
BOARD_MARGIN = 30
STONE_RADIUS = 12

# Window settings - side by side layout
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN
INFO_PANEL_WIDTH = 250
WINDOW_WIDTH = BOARD_WIDTH * 2 + INFO_PANEL_WIDTH + 40  # Two boards + center panel + margins
WINDOW_HEIGHT = BOARD_HEIGHT + 150  # Extra space for title and stats


class GameState:
    """Holds state for one game."""
    
    def __init__(self, agent1_name: str, agent2_name: str, board_offset_x: int):
        self.board = Board(BOARD_SIZE)
        self.agent1 = None
        self.agent2 = None
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.move_count = 0
        self.board_offset_x = board_offset_x
        self.thinking = False
        self.thinking_player = None
        
    def reset(self):
        """Reset game state."""
        self.board = Board(BOARD_SIZE)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.move_count = 0
        self.thinking = False
        self.thinking_player = None


class DualMatchGUI:
    """Main GUI class for dual match visualization."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Gomoku - Dual Agent Comparison")
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 40)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        
        # Game states - left and right
        self.left_game = GameState("Sklearn Agent", "Random Agent", 20)
        self.right_game = GameState("Minimax Agent", "Random Agent", 
                                   BOARD_WIDTH + INFO_PANEL_WIDTH + 20)
        
        # Statistics
        self.stats = {
            'sklearn': {'wins': 0, 'losses': 0, 'draws': 0, 'total_moves': 0},
            'minimax': {'wins': 0, 'losses': 0, 'draws': 0, 'total_moves': 0}
        }
        
        # Control state
        self.paused = False
        self.auto_play = False
        self.move_delay = 500  # milliseconds between moves
        self.last_move_time = 0
        
        # Model paths
        self.sklearn_model_path = self._find_sklearn_model()
        
        # Initialize agents
        self._initialize_agents()
        
        # Game control
        self.games_to_play = 1
        self.current_game_num = 1
        
    def _find_sklearn_model(self) -> Optional[str]:
        """Find sklearn model file."""
        possible_paths = [
            'checkpoints/sklearn_model.pkl',
            'checkpoints/gomoku_mlp_model.pkl',
            'checkpoints/best_sklearn_model.pkl'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _initialize_agents(self):
        """Initialize agents for both games."""
        # Left game: Sklearn vs Random
        try:
            if self.sklearn_model_path:
                self.left_game.agent1 = SklearnAgent(1, self.sklearn_model_path, temperature=1.0)
                print(f"✓ Loaded Sklearn model from {self.sklearn_model_path}")
            else:
                print("⚠️  No Sklearn model found, using Heuristic agent instead")
                from agents.heuristic_agent import HeuristicAgent
                self.left_game.agent1 = HeuristicAgent(1)
                self.left_game.agent1_name = "Heuristic Agent"
        except Exception as e:
            print(f"Failed to load Sklearn agent: {e}")
            from agents.heuristic_agent import HeuristicAgent
            self.left_game.agent1 = HeuristicAgent(1)
            self.left_game.agent1_name = "Heuristic Agent"
        
        self.left_game.agent2 = RandomAgent(2)
        
        # Right game: Minimax vs Random
        self.right_game.agent1 = MinimaxAgent(1, max_depth=2)
        self.right_game.agent2 = RandomAgent(2)
    
    def draw_board(self, game_state: GameState):
        """Draw a single board."""
        offset_x = game_state.board_offset_x
        offset_y = 100
        
        # Board background
        board_rect = pygame.Rect(
            offset_x - 10,
            offset_y - 10,
            BOARD_SIZE * CELL_SIZE + 20,
            BOARD_SIZE * CELL_SIZE + 20
        )
        pygame.draw.rect(self.screen, BROWN, board_rect)
        
        # Grid lines
        for i in range(BOARD_SIZE):
            # Vertical
            start_pos = (offset_x + i * CELL_SIZE, offset_y)
            end_pos = (offset_x + i * CELL_SIZE, offset_y + (BOARD_SIZE - 1) * CELL_SIZE)
            pygame.draw.line(self.screen, GRID_COLOR, start_pos, end_pos, 2)
            
            # Horizontal
            start_pos = (offset_x, offset_y + i * CELL_SIZE)
            end_pos = (offset_x + (BOARD_SIZE - 1) * CELL_SIZE, offset_y + i * CELL_SIZE)
            pygame.draw.line(self.screen, GRID_COLOR, start_pos, end_pos, 2)
        
        # Star points
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_points:
            x = offset_x + col * CELL_SIZE
            y = offset_y + row * CELL_SIZE
            pygame.draw.circle(self.screen, GRID_COLOR, (x, y), 4)
        
        # Draw stones
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if game_state.board.grid[row][col] != 0:
                    x = offset_x + col * CELL_SIZE
                    y = offset_y + row * CELL_SIZE
                    color = BLACK if game_state.board.grid[row][col] == 1 else WHITE
                    pygame.draw.circle(self.screen, color, (x, y), STONE_RADIUS)
                    pygame.draw.circle(self.screen, GRID_COLOR, (x, y), STONE_RADIUS, 2)
        
        # Highlight last move
        if game_state.last_move:
            row, col = game_state.last_move
            x = offset_x + col * CELL_SIZE
            y = offset_y + row * CELL_SIZE
            pygame.draw.circle(self.screen, LAST_MOVE_COLOR, (x, y), 5)
        
        # Highlight winning stones if game over
        if game_state.winner and game_state.last_move:
            self._highlight_winning_line(game_state, offset_x, offset_y)
    
    def _highlight_winning_line(self, game_state: GameState, offset_x: int, offset_y: int):
        """Highlight the winning five-in-a-row."""
        if not game_state.last_move:
            return
            
        row, col = game_state.last_move
        player = game_state.winner
        
        # Check all directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            stones = [(row, col)]
            
            # Check forward
            r, c = row + dr, col + dc
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   game_state.board.grid[r][c] == player):
                stones.append((r, c))
                r += dr
                c += dc
            
            # Check backward
            r, c = row - dr, col - dc
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   game_state.board.grid[r][c] == player):
                stones.append((r, c))
                r -= dr
                c -= dc
            
            if len(stones) >= 5:
                # Draw golden circles on winning stones
                for r, c in stones:
                    x = offset_x + c * CELL_SIZE
                    y = offset_y + r * CELL_SIZE
                    pygame.draw.circle(self.screen, WIN_HIGHLIGHT, (x, y), STONE_RADIUS + 3, 3)
                break
    
    def draw_game_info(self, game_state: GameState, y_position: int):
        """Draw game information below board."""
        offset_x = game_state.board_offset_x
        
        # Game title
        title = f"{game_state.agent1_name} vs {game_state.agent2_name}"
        title_surf = self.font_small.render(title, True, TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(offset_x + BOARD_SIZE * CELL_SIZE // 2, 70))
        self.screen.blit(title_surf, title_rect)
        
        # Current status
        if game_state.game_over:
            if game_state.winner:
                winner_name = game_state.agent1_name if game_state.winner == 1 else game_state.agent2_name
                status_text = f"Winner: {winner_name}"
                status_color = GREEN
            else:
                status_text = "Draw!"
                status_color = BLUE
        else:
            current_agent = game_state.agent1_name if game_state.current_player == 1 else game_state.agent2_name
            status_text = f"Playing: {current_agent}"
            status_color = BLUE
            
            if game_state.thinking:
                status_text += " (thinking...)"
        
        status_surf = self.font_small.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(center=(offset_x + BOARD_SIZE * CELL_SIZE // 2, y_position))
        self.screen.blit(status_surf, status_rect)
        
        # Move count
        moves_text = f"Moves: {game_state.move_count}"
        moves_surf = self.font_tiny.render(moves_text, True, TEXT_COLOR)
        moves_rect = moves_surf.get_rect(center=(offset_x + BOARD_SIZE * CELL_SIZE // 2, y_position + 25))
        self.screen.blit(moves_surf, moves_rect)
    
    def draw_center_panel(self):
        """Draw center information panel."""
        panel_x = BOARD_WIDTH + 20
        panel_y = 100
        panel_width = INFO_PANEL_WIDTH
        panel_height = BOARD_HEIGHT
        
        # Panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, GRAY, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, TEXT_COLOR, panel_rect, 2, border_radius=10)
        
        # Title
        title = self.font_medium.render("Statistics", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(panel_x + panel_width // 2, panel_y + 30))
        self.screen.blit(title, title_rect)
        
        # Sklearn stats
        y = panel_y + 70
        sklearn_title = self.font_small.render("Sklearn Agent", True, BLUE)
        self.screen.blit(sklearn_title, (panel_x + 20, y))
        y += 30
        
        sklearn_stats = self.stats['sklearn']
        total = sklearn_stats['wins'] + sklearn_stats['losses'] + sklearn_stats['draws']
        win_rate = (sklearn_stats['wins'] / total * 100) if total > 0 else 0
        
        stats_lines = [
            f"Wins: {sklearn_stats['wins']}",
            f"Losses: {sklearn_stats['losses']}",
            f"Draws: {sklearn_stats['draws']}",
            f"Win Rate: {win_rate:.1f}%",
            f"Avg Moves: {sklearn_stats['total_moves'] // max(total, 1)}"
        ]
        
        for line in stats_lines:
            text = self.font_tiny.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (panel_x + 30, y))
            y += 22
        
        # Separator
        y += 20
        pygame.draw.line(self.screen, TEXT_COLOR, 
                        (panel_x + 20, y), (panel_x + panel_width - 20, y), 2)
        y += 20
        
        # Minimax stats
        minimax_title = self.font_small.render("Minimax Agent", True, RED)
        self.screen.blit(minimax_title, (panel_x + 20, y))
        y += 30
        
        minimax_stats = self.stats['minimax']
        total = minimax_stats['wins'] + minimax_stats['losses'] + minimax_stats['draws']
        win_rate = (minimax_stats['wins'] / total * 100) if total > 0 else 0
        
        stats_lines = [
            f"Wins: {minimax_stats['wins']}",
            f"Losses: {minimax_stats['losses']}",
            f"Draws: {minimax_stats['draws']}",
            f"Win Rate: {win_rate:.1f}%",
            f"Avg Moves: {minimax_stats['total_moves'] // max(total, 1)}"
        ]
        
        for line in stats_lines:
            text = self.font_tiny.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (panel_x + 30, y))
            y += 22
        
        # Control instructions
        y = panel_y + panel_height - 150
        controls_title = self.font_small.render("Controls", True, TEXT_COLOR)
        self.screen.blit(controls_title, (panel_x + 20, y))
        y += 30
        
        controls = [
            "SPACE: Next Move",
            "A: Auto-play",
            "P: Pause",
            "R: New Games",
            "ESC: Quit"
        ]
        
        for control in controls:
            text = self.font_tiny.render(control, True, TEXT_COLOR)
            self.screen.blit(text, (panel_x + 20, y))
            y += 22
    
    def draw_title(self):
        """Draw main title."""
        title = self.font_large.render("Gomoku Agent Comparison", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Game counter
        subtitle = self.font_small.render(
            f"Game {self.current_game_num} / {self.games_to_play}", 
            True, TEXT_COLOR
        )
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 55))
        self.screen.blit(subtitle, subtitle_rect)
    
    def make_move(self, game_state: GameState):
        """Make one move in a game."""
        if game_state.game_over:
            return
        
        current_agent = (game_state.agent1 if game_state.current_player == 1 
                        else game_state.agent2)
        
        game_state.thinking = True
        game_state.thinking_player = game_state.current_player
        
        try:
            # Get move from agent
            row, col = current_agent.choose_move(game_state.board)
            
            # Make move
            if game_state.board.is_valid_move(row, col):
                game_state.board.make_move(row, col, game_state.current_player)
                game_state.last_move = (row, col)
                game_state.move_count += 1
                
                # Check for winner
                if game_state.board.check_winner(row, col, game_state.current_player):
                    game_state.winner = game_state.current_player
                    game_state.game_over = True
                elif game_state.board.is_full():
                    game_state.winner = None
                    game_state.game_over = True
                else:
                    game_state.current_player = 3 - game_state.current_player
        
        except Exception as e:
            print(f"Error during move: {e}")
            game_state.game_over = True
        
        finally:
            game_state.thinking = False
            game_state.thinking_player = None
    
    def update_statistics(self):
        """Update statistics after games end."""
        # Update sklearn stats
        if self.left_game.game_over:
            if self.left_game.winner == 1:
                self.stats['sklearn']['wins'] += 1
            elif self.left_game.winner == 2:
                self.stats['sklearn']['losses'] += 1
            else:
                self.stats['sklearn']['draws'] += 1
            self.stats['sklearn']['total_moves'] += self.left_game.move_count
        
        # Update minimax stats
        if self.right_game.game_over:
            if self.right_game.winner == 1:
                self.stats['minimax']['wins'] += 1
            elif self.right_game.winner == 2:
                self.stats['minimax']['losses'] += 1
            else:
                self.stats['minimax']['draws'] += 1
            self.stats['minimax']['total_moves'] += self.right_game.move_count
    
    def start_new_games(self):
        """Start new games."""
        self.left_game.reset()
        self.right_game.reset()
        self._initialize_agents()
        self.current_game_num += 1
        self.last_move_time = pygame.time.get_ticks()
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    elif event.key == pygame.K_SPACE:
                        # Manual step
                        if not self.left_game.game_over:
                            self.make_move(self.left_game)
                        if not self.right_game.game_over:
                            self.make_move(self.right_game)
                    
                    elif event.key == pygame.K_a:
                        # Toggle auto-play
                        self.auto_play = not self.auto_play
                        print(f"Auto-play: {'ON' if self.auto_play else 'OFF'}")
                    
                    elif event.key == pygame.K_p:
                        # Toggle pause
                        self.paused = not self.paused
                        print(f"Paused: {self.paused}")
                    
                    elif event.key == pygame.K_r:
                        # New games
                        if self.left_game.game_over and self.right_game.game_over:
                            self.update_statistics()
                            self.start_new_games()
            
            # Auto-play logic
            if self.auto_play and not self.paused:
                if current_time - self.last_move_time >= self.move_delay:
                    both_finished = self.left_game.game_over and self.right_game.game_over
                    
                    if both_finished:
                        # Both games finished, start new ones
                        self.update_statistics()
                        if self.current_game_num < self.games_to_play:
                            self.start_new_games()
                    else:
                        # Make moves in unfinished games
                        if not self.left_game.game_over:
                            self.make_move(self.left_game)
                        if not self.right_game.game_over:
                            self.make_move(self.right_game)
                        
                        self.last_move_time = current_time
            
            # Drawing
            self.screen.fill(WHITE)
            self.draw_title()
            self.draw_board(self.left_game)
            self.draw_board(self.right_game)
            self.draw_center_panel()
            self.draw_game_info(self.left_game, BOARD_HEIGHT + 120)
            self.draw_game_info(self.right_game, BOARD_HEIGHT + 120)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


def main():
    """Run the dual match GUI."""
    gui = DualMatchGUI()
    gui.run()


if __name__ == "__main__":
    main()
