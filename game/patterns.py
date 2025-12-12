import numpy as np
from typing import Dict, Tuple, List


class PatternDetector:
    PATTERN_SCORES = {
        'five': 100000,        # wining
        'open_four': 50000,    # _XXXX_
        'four': 10000,         # XXXX_ or |XXXX_
        'open_three': 5000,    # _XXX_
        'three': 1000,         # XXX_ or |XXX_
        'open_two': 500,       # _XX_
        'two': 100,            # XX_ or |XX_
        'one': 10,             # X  
    }
    
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        
    def evaluate_position(self, board_grid: np.ndarray, player: int) -> float:
        """
        Evaluate the board position for a given player.
        
        Args:
            board_grid: Current board state (numpy array)
            player: Player to evaluate for (1 or 2)
            
        Returns:
            Score for the position (positive = good for player)
        """
        opponent = 3 - player
        
        player_patterns = self.count_all_patterns(board_grid, player)
        opponent_patterns = self.count_all_patterns(board_grid, opponent)
        
        player_score = self._calculate_score(player_patterns)
        opponent_score = self._calculate_score(opponent_patterns)
        
        return player_score - opponent_score
    
    def count_all_patterns(self, board_grid: np.ndarray, player: int) -> Dict[str, int]:
        """
        Count all patterns for a player on the board.
        
        Args:
            board_grid: Current board state
            player: Player to count patterns for
            
        Returns:
            Dictionary of pattern counts
        """
        patterns = {
            'five': 0,
            'open_four': 0,
            'four': 0,
            'open_three': 0,
            'three': 0,
            'open_two': 0,
            'two': 0,
            'one': 0,
        }
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for direction in directions:
            dir_patterns = self._count_patterns_in_direction(
                board_grid, player, direction
            )
            for pattern_type, count in dir_patterns.items():
                patterns[pattern_type] += count
        
        return patterns
    
    def _count_patterns_in_direction(
        self, 
        board_grid: np.ndarray, 
        player: int, 
        direction: Tuple[int, int]
    ) -> Dict[str, int]:
        """
        Count patterns in a specific direction.
        
        Args:
            board_grid: Board state
            player: Player to check
            direction: Direction vector (dr, dc)
            
        Returns:
            Pattern counts for this direction
        """
        patterns = {
            'five': 0,
            'open_four': 0,
            'four': 0,
            'open_three': 0,
            'three': 0,
            'open_two': 0,
            'two': 0,
            'one': 0,
        }
        
        dr, dc = direction
        visited = set()
        
        # Scan all positions
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board_grid[row][col] == player and (row, col) not in visited:
                    # Found a stone, check the line
                    line = self._extract_line(board_grid, row, col, direction)
                    pattern = self._classify_pattern(line, player)
                    
                    if pattern:
                        patterns[pattern] += 1
                        
                        # Mark stones in this pattern as visited
                        length = self._get_consecutive_count(line, player)
                        for i in range(length):
                            new_row = row + i * dr
                            new_col = col + i * dc
                            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                                visited.add((new_row, new_col))
        
        return patterns
    
    def _extract_line(
        self, 
        board_grid: np.ndarray, 
        row: int, 
        col: int, 
        direction: Tuple[int, int],
        length: int = 9
    ) -> List[int]:
        """
        Extract a line of cells from a position in a direction.
        
        Args:
            board_grid: Board state
            row: Starting row
            col: Starting column
            direction: Direction vector
            length: Length of line to extract (default 9 for context)
            
        Returns:
            List of cell values (-1 for out of bounds, 0/1/2 for cells)
        """
        dr, dc = direction
        line = []
        
        # Go back a few steps to get context
        start_offset = -4
        
        for i in range(start_offset, length + start_offset):
            new_row = row + i * dr
            new_col = col + i * dc
            
            # Out of bounds = blocked (-1)
            if new_row < 0 or new_row >= self.board_size or \
               new_col < 0 or new_col >= self.board_size:
                line.append(-1)
            else:
                line.append(int(board_grid[new_row][new_col]))
        
        return line
    
    def _classify_pattern(self, line: List[int], player: int) -> str:
        """
        Classify a pattern in a line.
        
        Args:
            line: List of cell values
            player: Player to check
            
        Returns:
            Pattern type or empty string if no pattern
        """
        opponent = 3 - player
        
        # Convert line to string for pattern matching
        line_str = ''.join(['P' if x == player else 
                           'O' if x == opponent else 
                           'E' if x == 0 else 'B' for x in line])
        
        # Check for five (winning)
        if 'PPPPP' in line_str:
            return 'five'
        
        # Check for open four (_PPPP_)
        if 'EPPPPE' in line_str:
            return 'open_four'
        
        # Check for four (PPPP_ or _PPPP with block)
        if 'BPPPPE' in line_str or 'EPPPPB' in line_str or \
           'EPPPPO' in line_str or 'OPPPE' in line_str:
            return 'four'
        
        # Check for open three (_PPP_)
        if 'EPPPPE' in line_str:
            return 'open_three'
        if 'EPPEP' in line_str or 'PEPPPE' in line_str or 'EPPEPPE' in line_str:
            return 'open_three'
        
        # Check for three (PPP_ or _PPP with block)
        if 'BPPPPE' in line_str or 'EPPPB' in line_str or \
           'EPPPO' in line_str or 'OPPPE' in line_str:
            return 'three'
        if 'EPPEP' in line_str or 'PEPP' in line_str:
            return 'three'
        
        # Check for open two (_PP_)
        if 'EPPEE' in line_str or 'EEPPE' in line_str:
            return 'open_two'
        
        # Check for two (PP_ or _PP with block)
        if 'BPPE' in line_str or 'EPPB' in line_str or \
           'EPPO' in line_str or 'OPPE' in line_str:
            return 'two'
        
        # Single stone
        if 'P' in line_str:
            return 'one'
        
        return ''
    
    def _get_consecutive_count(self, line: List[int], player: int) -> int:
        """
        Get the count of consecutive stones in a line.
        
        Args:
            line: List of cell values
            player: Player to check
            
        Returns:
            Number of consecutive stones
        """
        count = 0
        for cell in line:
            if cell == player:
                count += 1
            elif count > 0:
                break
        return count
    
    def _calculate_score(self, patterns: Dict[str, int]) -> float:
        """
        Calculate total score from pattern counts.
        
        Args:
            patterns: Dictionary of pattern counts
            
        Returns:
            Total score
        """
        score = 0.0
        for pattern_type, count in patterns.items():
            score += self.PATTERN_SCORES[pattern_type] * count
        
        return score
    
    def get_threats(self, board_grid: np.ndarray, player: int) -> List[Tuple[int, int]]:
        """
        Find positions that create immediate threats (winning moves or blocks).
        
        Args:
            board_grid: Current board state
            player: Player to check threats for
            
        Returns:
            List of (row, col) positions that are threatening
        """
        threats = []
        
        # Check each empty position
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board_grid[row][col] == 0:
                    # Temporarily place stone
                    board_grid[row][col] = player
                    
                    # Check if this creates a winning pattern
                    patterns = self.count_all_patterns(board_grid, player)
                    
                    if patterns['five'] > 0 or patterns['open_four'] > 0:
                        threats.append((row, col))
                    
                    # Remove stone
                    board_grid[row][col] = 0
        
        return threats
