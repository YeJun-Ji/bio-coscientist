"""
Base Problem Handler - Abstract interface for problem-specific handlers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..core import ResearchGoal, Hypothesis


class BaseProblemHandler(ABC):
    """Base class for problem-type-specific research handlers"""
    
    def __init__(self, problem_type: str):
        self.problem_type = problem_type
    
    @abstractmethod
    def get_generation_prompt_additions(self, research_goal: ResearchGoal) -> str:
        """Get problem-specific additions to hypothesis generation prompts"""
        pass
    
    @abstractmethod
    def get_review_criteria(self, research_goal: ResearchGoal) -> Dict[str, Any]:
        """Get problem-specific review criteria"""
        pass
    
    @abstractmethod
    def validate_hypothesis(self, hypothesis: Hypothesis, research_goal: ResearchGoal) -> Dict[str, Any]:
        """Validate hypothesis meets problem-specific requirements"""
        pass
    
    @abstractmethod
    def get_expected_outputs(self, research_goal: ResearchGoal) -> List[str]:
        """Get list of expected outputs for this problem type"""
        pass
    
    def get_domain_knowledge(self) -> str:
        """Get relevant domain knowledge for this problem type"""
        return ""
