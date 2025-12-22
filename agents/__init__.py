"""
Agents Module - Multi-agent Sequential Confirmation Research System (v6.0)

The core agents for the RequirementAnswer-based workflow:
- ConfigurationAgent: Parses research problems into requirements
- GenerationAgent: Generates answers for requirements
- ReflectionCoachAgent: Provides qualitative feedback on 4 criteria
- TournamentRankingAgent: Ranks answers via pairwise comparison
- EvolutionArchitectAgent: Enriches confirmed answers (PASS in v6.0)
- SupervisorAgent: Orchestrates the Sequential Confirmation workflow

v6.0 Changes:
- Removed: LogVerificationAgent (Pre-check removed)
- Deprecated: QualityAssessmentAgent, RankingAgent, EvolutionAgent
"""

from .base_agent import BaseAgent
from .configuration_agent import ConfigurationAgent
from .generation_agent import GenerationAgent
from .reflection_coach_agent import ReflectionCoachAgent
from .tournament_ranking_agent import TournamentRankingAgent
from .evolution_architect_agent import EvolutionArchitectAgent
from .supervisor_agent import SupervisorAgent

# Deprecated agents (kept for backward compatibility)
from .quality_assessment_agent import QualityAssessmentAgent
from .ranking_agent import RankingAgent
from .evolution_agent import EvolutionAgent

__all__ = [
    # Core agents (v6.0)
    "BaseAgent",
    "ConfigurationAgent",
    "GenerationAgent",
    "ReflectionCoachAgent",
    "TournamentRankingAgent",
    "EvolutionArchitectAgent",
    "SupervisorAgent",
    # Deprecated (kept for compatibility)
    "QualityAssessmentAgent",
    "RankingAgent",
    "EvolutionAgent"
]
