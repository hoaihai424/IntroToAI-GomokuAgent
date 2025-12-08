from abc import ABC, abstractmethod
from typing import Tuple

class Agent(ABC):
    def __init__(self, player: int):
        self.player = player
    
    @abstractmethod
    def choose_move(self, board) -> Tuple[int, int]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass