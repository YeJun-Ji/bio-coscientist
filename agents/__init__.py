"""
Agents Module - Multi-agent research system
"""

from .base_agent import BaseAgent
from .generation_agent import GenerationAgent
from .reflection_agent import ReflectionAgent
from .ranking_agent import RankingAgent
from .proximity_agent import ProximityAgent
from .evolution_agent import EvolutionAgent
from .metareview_agent import MetaReviewAgent
from .supervisor_agent import SupervisorAgent

__all__ = [
    "BaseAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "RankingAgent",
    "ProximityAgent",
    "EvolutionAgent",
    "MetaReviewAgent",
    "SupervisorAgent"
]
