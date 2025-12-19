"""
Agents Module - Multi-agent Sequential Confirmation Research System

The core agents for the RequirementAnswer-based workflow:
- ConfigurationAgent: Parses research problems into requirements
- GenerationAgent: Generates answers for requirements
- ReflectionAgent: Reviews and evaluates answers
- RankingAgent: Ranks answers via ELO tournament
- EvolutionAgent: Evolves answers for improvement
- SupervisorAgent: Orchestrates the Sequential Confirmation workflow
"""

from .base_agent import BaseAgent
from .configuration_agent import ConfigurationAgent
from .generation_agent import GenerationAgent
from .reflection_agent import ReflectionAgent
from .ranking_agent import RankingAgent
from .evolution_agent import EvolutionAgent
from .supervisor_agent import SupervisorAgent

__all__ = [
    "BaseAgent",
    "ConfigurationAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "RankingAgent",
    "EvolutionAgent",
    "SupervisorAgent"
]
