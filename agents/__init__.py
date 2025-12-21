"""
Agents Module - Multi-agent Sequential Confirmation Research System (v4.0)

The core agents for the RequirementAnswer-based workflow:
- ConfigurationAgent: Parses research problems into requirements
- GenerationAgent: Generates answers for requirements
- LogVerificationAgent: Objective log-based verification (pre-check + verify)
- QualityAssessmentAgent: Domain-agnostic quality evaluation
- RankingAgent: Score-based answer ranking
- EvolutionAgent: Evolves answers for improvement
- SupervisorAgent: Orchestrates the Sequential Confirmation workflow
"""

from .base_agent import BaseAgent
from .configuration_agent import ConfigurationAgent
from .generation_agent import GenerationAgent
from .log_verification_agent import LogVerificationAgent
from .quality_assessment_agent import QualityAssessmentAgent
from .ranking_agent import RankingAgent
from .evolution_agent import EvolutionAgent
from .supervisor_agent import SupervisorAgent

__all__ = [
    "BaseAgent",
    "ConfigurationAgent",
    "GenerationAgent",
    "LogVerificationAgent",
    "QualityAssessmentAgent",
    "RankingAgent",
    "EvolutionAgent",
    "SupervisorAgent"
]
