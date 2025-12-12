"""
Pygame GUI for Gomoku game.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pygame
from typing import Optional, Tuple
from game.board import Board
from agents.base_agent import Agent
from agents.random_agent import RandomAgent
from agents.sklearn_agent import SklearnAgent
from agents.minimax import MinimaxAgent


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (205, 170, 125)
GRID_COLOR = (50, 50, 50)
HIGHLIGHT_COLOR = (255, 0, 0)
LAST_MOVE_COLOR = (0, 0, 255)
BUTTON_COLOR = (100, 150, 200)
BUTTON_HOVER_COLOR = (120, 170, 220)
TEXT_COLOR = (50, 50, 50)

# Board settings
BOARD_SIZE = 15
CELL_SIZE = 40
BOARD_MARGIN = 50
STONE_RADIUS = 16

# Window settings
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 300
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 100


class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.hovered = False
    
    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=5)
        
        text_surf = self.font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                return True
        return False


class HumanAgent(Agent):
    def __init__(self, player: int):
        super().__init__(player)
        self.pending_move = None
    
    def choose_move(self, board) -> Tuple[int, int]:
        while self.pending_move is None:
            pygame.time.wait(10)
        
        move = self.pending_move
        self.pending_move = None
        return move
    
    def set_move(self, row, col):
        """Set the human's chosen move."""
        self.pending_move = (row, col)
    
    def get_name(self) -> str:
        return f"Human_P{self.player}"


class GomokuGUI:
    """Main GUI class for Gomoku game."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Gomoku - Five in a Row")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game state
        self.board = Board(size=BOARD_SIZE)
        self.game_over = False
        self.winner = None
        self.current_player = 1
        self.last_move = None
        self.move_history = []
        
        # Agents
        self.agent1 = None
        self.agent2 = None
        
        # Batch game state
        self.batch_mode = False
        self.batch_games_total = 0
        self.batch_games_played = 0
        self.batch_stats = {'p1_wins': 0, 'p2_wins': 0, 'draws': 0, 'total_moves': 0}
        self.batch_complete = False  # Flag to show statistics
        self.input_mode = False  # For entering number of games
        self.input_text = ""
        
        # UI state
        self.agent_selection_mode = True
        self.buttons = self._create_agent_selection_buttons()
        
        # Selected agents (for setup)
        self.game_mode = None  # 'human1', 'human2', 'ai_vs_ai'
        self.selected_agent1_type = None  # AI type for player 1
        self.selected_agent2_type = None  # AI type for player 2
    
    def _create_agent_selection_buttons(self):
        """Create buttons for agent selection."""
        button_x = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 20
        button_y = 50
        button_width = 250
        button_height = 40
        button_gap = 10
        
        buttons = {
            'human1': Button(button_x, button_y, button_width, button_height, "Human vs AI", self.small_font),
            'ai_vs_ai': Button(button_x, button_y + button_height + button_gap, button_width, button_height, "AI vs AI", self.small_font),
            'human2': Button(button_x, button_y + 2*(button_height + button_gap), button_width, button_height, "Human vs Human", self.small_font),
        }
        
        # AI selection buttons (shown after choosing mode)
        ai_y = button_y + 3*(button_height + button_gap) + 30
        buttons['random'] = Button(button_x, ai_y, button_width, button_height, "Random AI", self.small_font)
        buttons['minimax'] = Button(button_x, ai_y + button_height + button_gap, button_width, button_height, "Minimax AI", self.small_font)
        buttons['ml'] = Button(button_x, ai_y + 2*(button_height + button_gap), button_width, button_height, "ML AI", self.small_font)
        
        # Control buttons
        control_y = WINDOW_HEIGHT - 100
        buttons['new_game'] = Button(button_x, control_y, 120, 40, "New Game", self.small_font)
        buttons['undo'] = Button(button_x + 130, control_y, 100, 40, "Undo", self.small_font)
        buttons['run_n_games'] = Button(button_x, control_y - 50, 230, 40, "Run N Games", self.small_font)
        
        return buttons
    
    def setup_agents(self):
        """Setup agents based on user selection."""
        if self.game_mode == 'human1':
            self.agent1 = HumanAgent(1)
            self.agent2 = self._create_ai_agent(2, self.selected_agent2_type)
        elif self.game_mode == 'human2':
            self.agent1 = HumanAgent(1)
            self.agent2 = HumanAgent(2)
        elif self.game_mode == 'ai_vs_ai':
            self.agent1 = self._create_ai_agent(1, self.selected_agent1_type)
            self.agent2 = self._create_ai_agent(2, self.selected_agent2_type)
        
        self.agent_selection_mode = False
    
    def _create_ai_agent(self, player: int, ai_type: str) -> Agent:
        """Create AI agent of specified type."""
        if ai_type == 'random':
            return RandomAgent(player)
        elif ai_type == 'minimax':
            return MinimaxAgent(player, depth=2)
        elif ai_type == 'ml':
            # Try to load sklearn model
            try:
                return SklearnAgent(player, 'checkpoints/sklearn_model.pkl', temperature=0.1)
            except:
                print("Sklearn model not found, using Random AI instead")
                return RandomAgent(player)
        else:
            return RandomAgent(player)
    
    def draw_board(self):
        """Draw the game board."""
        # Background
        self.screen.fill(WHITE)
        
        # Board background
        board_rect = pygame.Rect(
            BOARD_MARGIN - 20,
            BOARD_MARGIN - 20,
            BOARD_SIZE * CELL_SIZE + 40,
            BOARD_SIZE * CELL_SIZE + 40
        )
        pygame.draw.rect(self.screen, BROWN, board_rect)
        
        # Grid lines
        for i in range(BOARD_SIZE):
            # Vertical lines
            start_pos = (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN)
            end_pos = (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE)
            pygame.draw.line(self.screen, GRID_COLOR, start_pos, end_pos, 2)
            
            # Horizontal lines
            start_pos = (BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE)
            end_pos = (BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE)
            pygame.draw.line(self.screen, GRID_COLOR, start_pos, end_pos, 2)
        
        # Star points (optional, decorative)
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_points:
            x = BOARD_MARGIN + col * CELL_SIZE
            y = BOARD_MARGIN + row * CELL_SIZE
            pygame.draw.circle(self.screen, GRID_COLOR, (x, y), 5)
        
        # Draw stones
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board.grid[row][col] != 0:
                    x = BOARD_MARGIN + col * CELL_SIZE
                    y = BOARD_MARGIN + row * CELL_SIZE
                    color = BLACK if self.board.grid[row][col] == 1 else WHITE
                    pygame.draw.circle(self.screen, color, (x, y), STONE_RADIUS)
                    pygame.draw.circle(self.screen, GRID_COLOR, (x, y), STONE_RADIUS, 2)
        
        # Highlight last move
        if self.last_move:
            row, col = self.last_move
            x = BOARD_MARGIN + col * CELL_SIZE
            y = BOARD_MARGIN + row * CELL_SIZE
            pygame.draw.circle(self.screen, LAST_MOVE_COLOR, (x, y), 5)
    
    def draw_info(self):
        """Draw game information panel."""
        info_x = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 20
        
        # Title
        title = self.font.render("Gomoku", True, TEXT_COLOR)
        self.screen.blit(title, (info_x, 10))
        
        # Show batch complete statistics
        if self.batch_complete:
            self.draw_batch_statistics()
            return
        
        # Batch mode statistics
        if self.batch_mode:
            batch_title = self.small_font.render(f"Batch: {self.batch_games_played}/{self.batch_games_total}", True, TEXT_COLOR)
            self.screen.blit(batch_title, (info_x, 50))
            
            stats = self.batch_stats
            p1_wins_text = self.small_font.render(f"P1 Wins: {stats['p1_wins']}", True, TEXT_COLOR)
            p2_wins_text = self.small_font.render(f"P2 Wins: {stats['p2_wins']}", True, TEXT_COLOR)
            draws_text = self.small_font.render(f"Draws: {stats['draws']}", True, TEXT_COLOR)
            
            self.screen.blit(p1_wins_text, (info_x, 80))
            self.screen.blit(p2_wins_text, (info_x, 105))
            self.screen.blit(draws_text, (info_x, 130))
            
            if self.batch_games_played > 0:
                avg_moves = stats['total_moves'] / self.batch_games_played
                avg_text = self.small_font.render(f"Avg Moves: {avg_moves:.1f}", True, TEXT_COLOR)
                self.screen.blit(avg_text, (info_x, 155))
                
                # Win rates
                p1_rate = (stats['p1_wins'] / self.batch_games_played) * 100
                p2_rate = (stats['p2_wins'] / self.batch_games_played) * 100
                p1_rate_text = self.small_font.render(f"P1 Rate: {p1_rate:.1f}%", True, TEXT_COLOR)
                p2_rate_text = self.small_font.render(f"P2 Rate: {p2_rate:.1f}%", True, TEXT_COLOR)
                self.screen.blit(p1_rate_text, (info_x, 180))
                self.screen.blit(p2_rate_text, (info_x, 205))
        
        # Current player
        y_offset = 250 if not self.batch_mode else 230
        if not self.game_over:
            player_text = f"Player {self.current_player}'s turn"
            color_text = "Black" if self.current_player == 1 else "White"
            player_surf = self.small_font.render(player_text, True, TEXT_COLOR)
            color_surf = self.small_font.render(f"({color_text})", True, TEXT_COLOR)
            self.screen.blit(player_surf, (info_x, y_offset))
            self.screen.blit(color_surf, (info_x, y_offset + 30))
        else:
            if self.winner:
                winner_text = f"Player {self.winner} wins!"
                color_text = "Black" if self.winner == 1 else "White"
            else:
                winner_text = "Draw!"
                color_text = ""
            
            winner_surf = self.font.render(winner_text, True, (200, 0, 0))
            self.screen.blit(winner_surf, (info_x, y_offset))
            if color_text:
                color_surf = self.small_font.render(f"({color_text})", True, TEXT_COLOR)
                self.screen.blit(color_surf, (info_x, y_offset + 35))
        
        # Move count
        y_offset = 320 if not self.batch_mode else 360
        move_text = f"Moves: {len(self.move_history)}"
        move_surf = self.small_font.render(move_text, True, TEXT_COLOR)
        self.screen.blit(move_surf, (info_x, y_offset))
        
        # Agent info
        y_offset = 400 if not self.batch_mode else 400
        if self.agent1 and self.agent2:
            agent1_text = f"P1: {self.agent1.get_name()}"
            agent2_text = f"P2: {self.agent2.get_name()}"
            self.screen.blit(self.small_font.render(agent1_text, True, TEXT_COLOR), (info_x, y_offset))
            self.screen.blit(self.small_font.render(agent2_text, True, TEXT_COLOR), (info_x, y_offset + 30))
    
    def draw_batch_statistics(self):
        """Draw final batch statistics in a nice panel."""
        stats = self.batch_stats
        total = self.batch_games_played
        
        info_x = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 20
        panel_y = 50
        
        # Draw panel background
        panel_width = 260
        panel_height = 450
        panel_rect = pygame.Rect(info_x - 10, panel_y - 10, panel_width, panel_height)
        pygame.draw.rect(self.screen, (240, 245, 255), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (50, 100, 200), panel_rect, 3, border_radius=10)
        
        # Title
        title_text = self.font.render("Final Results", True, (50, 100, 200))
        self.screen.blit(title_text, (info_x, panel_y))
        
        y = panel_y + 50
        
        # Total games
        games_text = self.small_font.render(f"Total Games: {total}", True, TEXT_COLOR)
        self.screen.blit(games_text, (info_x, y))
        
        # Separator
        y += 40
        pygame.draw.line(self.screen, (100, 100, 100), (info_x, y), (info_x + 230, y), 2)
        y += 20
        
        # Player 1 stats
        p1_name = self.agent1.get_name() if self.agent1 else "Player 1"
        p1_text = self.small_font.render(f"Player 1 (Black):", True, TEXT_COLOR)
        self.screen.blit(p1_text, (info_x, y))
        y += 25
        p1_name_text = self.small_font.render(p1_name, True, (100, 100, 100))
        self.screen.blit(p1_name_text, (info_x + 10, y))
        y += 25
        
        p1_wins_pct = (stats['p1_wins']/total*100) if total > 0 else 0
        p1_wins_text = self.small_font.render(f"Wins: {stats['p1_wins']} ({p1_wins_pct:.1f}%)", True, (0, 150, 0))
        self.screen.blit(p1_wins_text, (info_x + 10, y))
        
        # Separator
        y += 40
        pygame.draw.line(self.screen, (100, 100, 100), (info_x, y), (info_x + 230, y), 2)
        y += 20
        
        # Player 2 stats
        p2_name = self.agent2.get_name() if self.agent2 else "Player 2"
        p2_text = self.small_font.render(f"Player 2 (White):", True, TEXT_COLOR)
        self.screen.blit(p2_text, (info_x, y))
        y += 25
        p2_name_text = self.small_font.render(p2_name, True, (100, 100, 100))
        self.screen.blit(p2_name_text, (info_x + 10, y))
        y += 25
        
        p2_wins_pct = (stats['p2_wins']/total*100) if total > 0 else 0
        p2_wins_text = self.small_font.render(f"Wins: {stats['p2_wins']} ({p2_wins_pct:.1f}%)", True, (0, 150, 0))
        self.screen.blit(p2_wins_text, (info_x + 10, y))
        
        # Separator
        y += 40
        pygame.draw.line(self.screen, (100, 100, 100), (info_x, y), (info_x + 230, y), 2)
        y += 20
        
        # Draws
        draws_pct = (stats['draws']/total*100) if total > 0 else 0
        draws_text = self.small_font.render(f"Draws: {stats['draws']} ({draws_pct:.1f}%)", True, TEXT_COLOR)
        self.screen.blit(draws_text, (info_x, y))
        
        # Average moves
        y += 35
        avg_moves = stats['total_moves']/total if total > 0 else 0
        avg_text = self.small_font.render(f"Avg Moves: {avg_moves:.1f}", True, TEXT_COLOR)
        self.screen.blit(avg_text, (info_x, y))
        
        # Instruction
        y += 50
        instruction_text = self.small_font.render("Click 'New Game'", True, (100, 100, 100))
        instruction_text2 = self.small_font.render("to continue", True, (100, 100, 100))
        self.screen.blit(instruction_text, (info_x + 40, y))
        self.screen.blit(instruction_text2, (info_x + 50, y + 20))
    
    def draw_agent_selection(self):
        """Draw agent selection screen."""
        info_x = BOARD_SIZE * CELL_SIZE + 2 * BOARD_MARGIN + 20
        
        # Title
        if not self.game_mode:
            title = self.font.render("Select Game Mode", True, TEXT_COLOR)
            self.screen.blit(title, (50, 20))
            
            # Draw mode buttons
            self.buttons['human1'].draw(self.screen)
            self.buttons['ai_vs_ai'].draw(self.screen)
            self.buttons['human2'].draw(self.screen)
        
        # If human vs AI mode selected, show AI selection for Player 2
        elif self.game_mode == 'human1' and not self.selected_agent2_type:
            title = self.small_font.render("Select AI for Player 2:", True, TEXT_COLOR)
            self.screen.blit(title, (info_x, 200))
            
            self.buttons['random'].draw(self.screen)
            self.buttons['minimax'].draw(self.screen)
            self.buttons['ml'].draw(self.screen)
        
        # If AI vs AI mode selected, show AI selection for both players
        elif self.game_mode == 'ai_vs_ai':
            if not self.selected_agent1_type:
                title = self.small_font.render("Select AI for Player 1 (Black):", True, TEXT_COLOR)
                self.screen.blit(title, (info_x, 200))
                
                self.buttons['random'].draw(self.screen)
                self.buttons['minimax'].draw(self.screen)
                self.buttons['ml'].draw(self.screen)
            elif not self.selected_agent2_type:
                title = self.small_font.render("Select AI for Player 2 (White):", True, TEXT_COLOR)
                self.screen.blit(title, (info_x, 200))
                
                # Show selected Player 1 agent
                p1_text = self.small_font.render(f"Player 1: {self.selected_agent1_type.upper()}", True, (0, 150, 0))
                self.screen.blit(p1_text, (info_x, 170))
                
                self.buttons['random'].draw(self.screen)
                self.buttons['minimax'].draw(self.screen)
                self.buttons['ml'].draw(self.screen)
    
    def draw_buttons(self):
        """Draw control buttons."""
        self.buttons['new_game'].draw(self.screen)
        self.buttons['undo'].draw(self.screen)
        
        # Only show "Run N Games" button in AI vs AI mode
        if self.game_mode == 'ai_vs_ai' and not self.batch_mode and not self.batch_complete:
            self.buttons['run_n_games'].draw(self.screen)
        
        # Draw input box if in input mode (centered on screen)
        if self.input_mode:
            # Semi-transparent overlay
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))
            
            # Dialog box
            dialog_width = 400
            dialog_height = 200
            dialog_x = (WINDOW_WIDTH - dialog_width) // 2
            dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
            
            dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
            pygame.draw.rect(self.screen, WHITE, dialog_rect, border_radius=10)
            pygame.draw.rect(self.screen, BLACK, dialog_rect, 3, border_radius=10)
            
            # Title
            title_text = self.font.render("Run Multiple Games", True, TEXT_COLOR)
            title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, dialog_y + 30))
            self.screen.blit(title_text, title_rect)
            
            # Prompt text
            prompt_text = self.small_font.render("Enter number of games to run:", True, TEXT_COLOR)
            prompt_rect = prompt_text.get_rect(center=(WINDOW_WIDTH // 2, dialog_y + 70))
            self.screen.blit(prompt_text, prompt_rect)
            
            # Input box
            input_box_width = 200
            input_box_x = (WINDOW_WIDTH - input_box_width) // 2
            input_box_y = dialog_y + 100
            input_rect = pygame.Rect(input_box_x, input_box_y, input_box_width, 40)
            pygame.draw.rect(self.screen, (240, 240, 240), input_rect)
            pygame.draw.rect(self.screen, BLACK, input_rect, 2)
            
            # Input text
            input_surface = self.small_font.render(self.input_text, True, BLACK)
            self.screen.blit(input_surface, (input_box_x + 10, input_box_y + 10))
            
            # Instructions
            instruction_text = self.small_font.render("Press ENTER to start, ESC to cancel", True, TEXT_COLOR)
            instruction_rect = instruction_text.get_rect(center=(WINDOW_WIDTH // 2, dialog_y + 160))
            self.screen.blit(instruction_text, instruction_rect)
    
    def get_board_position(self, mouse_pos) -> Optional[Tuple[int, int]]:
        """Convert mouse position to board coordinates."""
        x, y = mouse_pos
        
        # Check if click is within board bounds
        if (BOARD_MARGIN - CELL_SIZE//2 <= x <= BOARD_MARGIN + (BOARD_SIZE-1) * CELL_SIZE + CELL_SIZE//2 and
            BOARD_MARGIN - CELL_SIZE//2 <= y <= BOARD_MARGIN + (BOARD_SIZE-1) * CELL_SIZE + CELL_SIZE//2):
            
            # Find nearest intersection
            col = round((x - BOARD_MARGIN) / CELL_SIZE)
            row = round((y - BOARD_MARGIN) / CELL_SIZE)
            
            # Validate position
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return (row, col)
        
        return None
    
    def handle_click(self, pos):
        """Handle mouse click."""
        if self.agent_selection_mode:
            # Handle game mode selection
            if not self.game_mode:
                if self.buttons['human1'].rect.collidepoint(pos):
                    self.game_mode = 'human1'
                elif self.buttons['human2'].rect.collidepoint(pos):
                    self.game_mode = 'human2'
                    self.setup_agents()
                elif self.buttons['ai_vs_ai'].rect.collidepoint(pos):
                    self.game_mode = 'ai_vs_ai'
            
            # Handle AI type selection
            elif self.game_mode == 'human1' and not self.selected_agent2_type:
                for ai_type in ['random', 'minimax', 'ml']:
                    if self.buttons[ai_type].rect.collidepoint(pos):
                        self.selected_agent2_type = ai_type
                        self.setup_agents()
            
            elif self.game_mode == 'ai_vs_ai':
                if not self.selected_agent1_type:
                    for ai_type in ['random', 'minimax', 'ml']:
                        if self.buttons[ai_type].rect.collidepoint(pos):
                            self.selected_agent1_type = ai_type
                elif not self.selected_agent2_type:
                    for ai_type in ['random', 'minimax', 'ml']:
                        if self.buttons[ai_type].rect.collidepoint(pos):
                            self.selected_agent2_type = ai_type
                            self.setup_agents()
            
            return
        
        # Handle "Run N Games" button
        if self.game_mode == 'ai_vs_ai' and not self.batch_mode:
            if self.buttons['run_n_games'].handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': pos})):
                self.input_mode = True
                self.input_text = ""
                return
        
        # Handle control buttons
        if self.buttons['new_game'].handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': pos})):
            self.new_game()
            return
        
        if self.buttons['undo'].handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': pos})):
            self.undo_move()
            return
        
        # Handle board click
        if not self.game_over:
            board_pos = self.get_board_position(pos)
            if board_pos:
                row, col = board_pos
                current_agent = self.agent1 if self.current_player == 1 else self.agent2
                
                # Only allow human players to click
                if isinstance(current_agent, HumanAgent):
                    if self.board.is_valid_move(row, col):
                        self.make_move(row, col)
    
    def make_move(self, row, col):
        """Make a move on the board."""
        if self.board.make_move(row, col, self.current_player):
            self.last_move = (row, col)
            self.move_history.append((row, col, self.current_player))
            
            # Check for winner
            if self.board.check_winner(row, col, self.current_player):
                self.game_over = True
                self.winner = self.current_player
                if self.batch_mode:
                    self.batch_stats['p1_wins' if self.winner == 1 else 'p2_wins'] += 1
                    self.batch_stats['total_moves'] += len(self.move_history)
            elif len(self.board.get_legal_moves()) == 0:
                self.game_over = True
                self.winner = None
                if self.batch_mode:
                    self.batch_stats['draws'] += 1
                    self.batch_stats['total_moves'] += len(self.move_history)
            else:
                self.current_player = 3 - self.current_player
    
    def undo_move(self):
        """Undo last move."""
        if len(self.move_history) > 0 and not self.batch_mode:
            self.board.undo_move()
            self.move_history.pop()
            
            if len(self.move_history) > 0:
                self.last_move = self.move_history[-1][:2]
            else:
                self.last_move = None
            
            self.current_player = 3 - self.current_player
            self.game_over = False
            self.winner = None
    
    def new_game(self):
        """Start a new game."""
        # Reset batch complete flag
        if self.batch_complete:
            self.batch_complete = False
            self.batch_stats = {'p1_wins': 0, 'p2_wins': 0, 'draws': 0, 'total_moves': 0}
        
        if self.batch_mode:
            # In batch mode, start next game automatically
            self.batch_games_played += 1
            if self.batch_games_played < self.batch_games_total:
                self.board = Board(size=BOARD_SIZE)
                self.game_over = False
                self.winner = None
                self.current_player = 1
                self.last_move = None
                self.move_history = []
            else:
                # Batch complete - show final statistics
                self.print_batch_statistics()
                self.batch_mode = False
                self.batch_complete = True
        else:
            # Normal new game
            self.board = Board(size=BOARD_SIZE)
            self.game_over = False
            self.winner = None
            self.current_player = 1
            self.last_move = None
            self.move_history = []
            self.agent_selection_mode = True
            self.game_mode = None
            self.selected_agent1_type = None
            self.selected_agent2_type = None
    
    def start_batch_mode(self, num_games: int):
        """Start batch mode to run N consecutive games."""
        self.batch_mode = True
        self.batch_games_total = num_games
        self.batch_games_played = 0
        self.batch_stats = {'p1_wins': 0, 'p2_wins': 0, 'draws': 0, 'total_moves': 0}
        self.input_mode = False
        
        # Reset the board for first game
        self.board = Board(size=BOARD_SIZE)
        self.game_over = False
        self.winner = None
        self.current_player = 1
        self.last_move = None
        self.move_history = []
        
        print(f"\n{'='*60}")
        print(f"Starting batch mode: {num_games} games")
        print(f"Player 1: {self.agent1.get_name()}")
        print(f"Player 2: {self.agent2.get_name()}")
        print(f"{'='*60}\n")
    
    def print_batch_statistics(self):
        """Print final batch statistics to console."""
        stats = self.batch_stats
        total = self.batch_games_played
        
        print(f"\n{'='*60}")
        print(f"BATCH STATISTICS - {total} games completed")
        print(f"{'='*60}")
        print(f"Player 1 ({self.agent1.get_name()}) wins: {stats['p1_wins']} ({stats['p1_wins']/total*100:.1f}%)")
        print(f"Player 2 ({self.agent2.get_name()}) wins: {stats['p2_wins']} ({stats['p2_wins']/total*100:.1f}%)")
        print(f"Draws: {stats['draws']} ({stats['draws']/total*100:.1f}%)")
        print(f"Average moves per game: {stats['total_moves']/total:.1f}")
        print(f"{'='*60}\n")
    
    def ai_move(self):
        """Let AI make a move."""
        current_agent = self.agent1 if self.current_player == 1 else self.agent2
        
        if not isinstance(current_agent, HumanAgent) and not self.game_over:
            row, col = current_agent.choose_move(self.board)
            self.make_move(row, col)
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle text input
                if event.type == pygame.KEYDOWN and self.input_mode:
                    if event.key == pygame.K_RETURN:
                        # Start batch mode if valid number entered
                        try:
                            num_games = int(self.input_text)
                            if num_games > 0:
                                self.start_batch_mode(num_games)
                        except ValueError:
                            print("Invalid number entered")
                        self.input_mode = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.input_mode = False
                    elif event.unicode.isdigit():
                        self.input_text += event.unicode
                
                # Handle button hover
                for button in self.buttons.values():
                    button.handle_event(event)
                
                if event.type == pygame.MOUSEBUTTONDOWN and not self.input_mode:
                    self.handle_click(event.pos)
            
            # Clear screen
            self.screen.fill(WHITE)
            
            if self.agent_selection_mode:
                self.draw_agent_selection()
            else:
                self.draw_board()
                self.draw_info()
                self.draw_buttons()
                
                # AI move (non-blocking for visualization)
                if not self.game_over:
                    current_agent = self.agent1 if self.current_player == 1 else self.agent2
                    if not isinstance(current_agent, HumanAgent):
                        # Faster delay in batch mode
                        if self.batch_mode:
                            pygame.time.wait(10)
                        else:
                            pygame.time.wait(300)
                        self.ai_move()
                elif self.batch_mode and self.game_over:
                    # Auto-start next game in batch mode
                    pygame.time.wait(50)
                    self.new_game()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    gui = GomokuGUI()
    gui.run()
