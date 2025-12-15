"""
Core data structures and enumerations for Bio AI Co-Scientist
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class HypothesisStatus(Enum):
    """Status of hypothesis in the research pipeline"""
    GENERATED = "generated"
    UNDER_REVIEW = "under_review"
    PASSED_REVIEW = "passed_review"
    FAILED_REVIEW = "failed_review"
    REJECTED = "rejected"  # Failed review
    IN_TOURNAMENT = "in_tournament"  # Ready for ranking
    ACTIVE = "active"  # Active in research cycle
    EVOLVED = "evolved"
    ARCHIVED = "archived"


class ProblemType(Enum):
    """Types of bio AI research problems supported"""
    GENE_SIMILARITY = "gene_function_similarity"  # Problem 1
    RNA_STABILITY = "rna_stability_mechanism"     # Problem 2
    PROTEIN_BINDER = "protein_binder_design"       # Problem 3
    TARGET_DISCOVERY = "therapeutic_target_discovery"  # Problem 4
    DRUG_REPOSITIONING = "drug_repositioning"     # Problem 5


@dataclass
class Hypothesis:
    """Represents a scientific hypothesis or research proposal"""
    id: str
    content: str
    category: str
    summary: str
    generated_at: datetime
    status: HypothesisStatus = HypothesisStatus.GENERATED
    
    # Evaluation scores
    elo_rating: float = 1200.0
    correctness_score: float = 0.0
    quality_score: float = 0.0
    novelty_score: float = 0.0
    testability_score: float = 0.0
    
    # Reviews and feedback
    initial_review: Optional[Dict] = None
    full_review: Optional[Dict] = None
    deep_verification: Optional[Dict] = None
    observation_review: Optional[Dict] = None
    simulation_review: Optional[Dict] = None
    
    # Tournament history
    tournament_matches: List[Dict] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    
    # Evolution tracking
    parent_ids: List[str] = field(default_factory=list)
    evolution_method: Optional[str] = None
    iteration: int = 1
    
    # Literature references
    supporting_papers: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Review:
    """Represents a review of a hypothesis"""
    review_id: str
    hypothesis_id: str
    review_type: str  # initial, full, deep_verification, observation, simulation, tournament
    reviewer: str
    timestamp: datetime
    
    # Review content
    correctness_assessment: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    novelty_assessment: Dict[str, Any]
    
    # Detailed feedback
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    
    # Decision
    pass_review: bool
    confidence: float
    
    # Supporting evidence
    literature_references: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)


@dataclass
class ResearchGoal:
    """Represents the overarching research goal"""
    goal_id: str
    description: str
    domain: str
    focus_areas: List[str]
    constraints: Dict[str, Any]
    success_criteria: List[str]
    created_at: datetime
    problem_type: Optional[ProblemType] = None  # Specific problem type


@dataclass
class TournamentMatch:
    """Represents a tournament match between two hypotheses"""
    match_id: str
    hypothesis_a_id: str
    hypothesis_b_id: str
    timestamp: datetime
    winner_id: str
    decision_rationale: str
    
    # Debate rounds
    debate_rounds: List[Dict] = field(default_factory=list)
    
    # Result
    elo_changes: Dict[str, float] = field(default_factory=dict)


@dataclass
class Paper:
    """Represents a scientific paper"""
    title: str
    abstract: str
    authors: List[str]
    year: int
    source: str  # "pubmed", "semantic_scholar", "arxiv"
    url: str
    citations: int = 0
    id: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
