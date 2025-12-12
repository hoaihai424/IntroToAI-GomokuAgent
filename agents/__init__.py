from agents.base_agent import Agent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent

# Try to import sklearn agent (optional)
try:
    from agents.sklearn_agent import SklearnAgent
    __all__ = ['Agent', 'RandomAgent', 'HeuristicAgent', 'SklearnAgent']
except ImportError:
    __all__ = ['Agent', 'RandomAgent', 'HeuristicAgent']