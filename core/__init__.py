"""
Core data structures for Bio AI Co-Scientist Sequential Confirmation Architecture

This module contains the essential data structures:
- Requirement: Parsed research requirements
- RequirementAnswer: Answers to requirements (top-level evaluation unit)
- ParsedProblem: Complete parsed research problem
- ResearchGoal: Research goal metadata
- Paper: Literature reference
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================================
# Requirement-Based Problem Parsing Structures
# ============================================================================

@dataclass
class Requirement:
    """
    Represents a requirement that must be answered.
    Extracted from problem statements - each (1), (2), etc. becomes a requirement.

    Supports both flat (1, 2, 3) and hierarchical (A.1, A.2, B.1) structures.
    """
    requirement_id: str                    # "1", "2", "A", "A.1", "B.2"
    title: str                             # Short title of the requirement
    description: str                       # Full description of what needs to be answered
    parent_id: Optional[str] = None        # "A" for "A.1", None for top-level
    expected_deliverables: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # Requirement IDs this depends on
    can_parallelize: bool = True           # Can answer in parallel with others
    order: int = 0                         # Order within parent
    requirement_type: str = "answer"       # answer, design, analysis, validation
    priority: int = 1                      # 1=required, 2=recommended, 3=optional

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "title": self.title,
            "description": self.description,
            "parent_id": self.parent_id,
            "expected_deliverables": self.expected_deliverables,
            "depends_on": self.depends_on,
            "can_parallelize": self.can_parallelize,
            "order": self.order,
            "requirement_type": self.requirement_type,
            "priority": self.priority,
            # Backward compatibility fields
            "step_id": self.requirement_id,
            "required_deliverables": self.expected_deliverables,
            "step_type": self.requirement_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Requirement":
        return cls(
            requirement_id=data.get("requirement_id", data.get("step_id", "")),
            title=data.get("title", ""),
            description=data.get("description", ""),
            parent_id=data.get("parent_id"),
            expected_deliverables=data.get("expected_deliverables", data.get("required_deliverables", [])),
            depends_on=data.get("depends_on", []),
            can_parallelize=data.get("can_parallelize", True),
            order=data.get("order", 0),
            requirement_type=data.get("requirement_type", data.get("step_type", "answer")),
            priority=data.get("priority", 1)
        )

    # Backward compatibility properties
    @property
    def step_id(self) -> str:
        return self.requirement_id

    @property
    def required_deliverables(self) -> List[str]:
        return self.expected_deliverables

    @property
    def step_type(self) -> str:
        return self.requirement_type


# Backward compatibility alias
ResearchStep = Requirement


@dataclass
class RequirementAnswer:
    """
    Represents an answer to a specific requirement.

    RequirementAnswer is the top-level evaluation unit in Sequential Confirmation:
    - Multiple answers can be generated for each requirement
    - Each answer is independently evaluated, ranked, and evolved
    - Status progresses: generated → reviewed → ranked → confirmed

    Key feature: answers can reference previous answers via 'builds_on' field,
    enabling cumulative, coherent answer generation.
    """
    # Core identification
    id: str = ""                           # Unique answer ID
    requirement_id: str = ""               # Matching Requirement.requirement_id
    requirement_title: str = ""            # Copy of requirement title for convenience

    # Answer content
    answer: str = ""                       # The actual answer content
    rationale: str = ""                    # Why this answer was chosen
    deliverables: Dict[str, Any] = field(default_factory=dict)  # Produced outputs
    confidence: float = 0.5                # 0.0 to 1.0
    builds_on: List[str] = field(default_factory=list)  # IDs of previous answers this builds upon

    # Ranking Support (ELO Tournament)
    elo_rating: float = 1200.0             # ELO rating for ranking
    wins: int = 0                          # Tournament wins
    losses: int = 0                        # Tournament losses

    # Status Management (Sequential Confirmation)
    status: str = "generated"
    # Status flow: generated → reviewed → ranked → confirmed

    # Review Results (from ReflectionAgent)
    review: Optional[Dict[str, Any]] = None  # Full review (legacy - kept for backward compatibility)
    quality_score: float = 0.0             # Overall quality score (0.0 to 1.0) - backward compatibility
    novelty_score: float = 0.0             # Novelty score (0.0 to 1.0) - backward compatibility

    # Data-Driven Evaluation (v3.0)
    observation_score: float = 0.0         # Score from observation_review (0.0 to 1.0)
    simulation_score: float = 0.0          # Score from simulation_review (0.0 to 1.0)
    observation_review: Optional[Dict[str, Any]] = None  # Full observation review results
    simulation_review: Optional[Dict[str, Any]] = None   # Full simulation review results

    # Evolution Tracking
    parent_ids: List[str] = field(default_factory=list)  # IDs of parent answers
    evolution_method: Optional[str] = None  # grounding, coherence, simplification, divergent
    iteration: int = 1                     # Evolution iteration number

    # Log-Based Evaluation (v4.0)
    verification_score: float = 0.0        # From LogVerificationAgent (objective, fact-checked against logs)
    composite_score: float = 0.0           # verification_score*0.5 + quality_score*0.5

    # Tournament Ranking (v5.0)
    tournament_rank: int = 0                # Final rank from tournament (1=best, 2=second, etc.)
    tournament_matchups: List[Dict] = field(default_factory=list)  # All matchups this answer participated in

    # Metadata
    generated_at: Optional[datetime] = None
    generation_method: str = "data_based"  # data_based, assumption, expansion, evolution
    data_sources: List[str] = field(default_factory=list)  # Which MCP tools/data sources were used
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.id and self.requirement_id:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            self.id = f"ans_{self.requirement_id}_{timestamp}"
        if self.generated_at is None:
            self.generated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Core
            "id": self.id,
            "requirement_id": self.requirement_id,
            "requirement_title": self.requirement_title,
            # Content
            "answer": self.answer,
            "rationale": self.rationale,
            "deliverables": self.deliverables,
            "confidence": self.confidence,
            "builds_on": self.builds_on,
            # Ranking
            "elo_rating": self.elo_rating,
            "wins": self.wins,
            "losses": self.losses,
            # Status
            "status": self.status,
            "review": self.review,
            "quality_score": self.quality_score,
            "novelty_score": self.novelty_score,
            # Data-Driven Evaluation (v3.0)
            "observation_score": self.observation_score,
            "simulation_score": self.simulation_score,
            "observation_review": self.observation_review,
            "simulation_review": self.simulation_review,
            # Evolution
            "parent_ids": self.parent_ids,
            "evolution_method": self.evolution_method,
            "iteration": self.iteration,
            # Log-Based Evaluation (v4.0)
            "verification_score": self.verification_score,
            "composite_score": self.composite_score,
            # Tournament Ranking (v5.0)
            "tournament_rank": self.tournament_rank,
            "tournament_matchups": self.tournament_matchups,
            # Metadata
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "generation_method": self.generation_method,
            "data_sources": self.data_sources,
            "metadata": self.metadata,
            # Backward compatibility fields
            "step_id": self.requirement_id,
            "step_title": self.requirement_title
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementAnswer":
        # Parse generated_at
        generated_at = data.get("generated_at")
        if isinstance(generated_at, str):
            try:
                generated_at = datetime.fromisoformat(generated_at)
            except ValueError:
                generated_at = datetime.now()
        elif generated_at is None:
            generated_at = datetime.now()

        return cls(
            # Core
            id=data.get("id", ""),
            requirement_id=data.get("requirement_id", data.get("step_id", "")),
            requirement_title=data.get("requirement_title", data.get("step_title", "")),
            # Content
            answer=data.get("answer", ""),
            rationale=data.get("rationale", ""),
            deliverables=data.get("deliverables", {}),
            confidence=data.get("confidence", 0.5),
            builds_on=data.get("builds_on", []),
            # Ranking
            elo_rating=data.get("elo_rating", 1200.0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            # Status
            status=data.get("status", "generated"),
            review=data.get("review"),
            quality_score=data.get("quality_score", 0.0),
            novelty_score=data.get("novelty_score", 0.0),
            # Data-Driven Evaluation (v3.0)
            observation_score=data.get("observation_score", 0.0),
            simulation_score=data.get("simulation_score", 0.0),
            observation_review=data.get("observation_review"),
            simulation_review=data.get("simulation_review"),
            # Evolution
            parent_ids=data.get("parent_ids", []),
            evolution_method=data.get("evolution_method"),
            iteration=data.get("iteration", 1),
            # Log-Based Evaluation (v4.0)
            verification_score=data.get("verification_score", 0.0),
            composite_score=data.get("composite_score", 0.0),
            # Tournament Ranking (v5.0)
            tournament_rank=data.get("tournament_rank", 0),
            tournament_matchups=data.get("tournament_matchups", []),
            # Metadata
            generated_at=generated_at,
            generation_method=data.get("generation_method", "data_based"),
            data_sources=data.get("data_sources", []),
            metadata=data.get("metadata", {})
        )

    # Backward compatibility properties
    @property
    def step_id(self) -> str:
        return self.requirement_id

    @property
    def step_title(self) -> str:
        return self.requirement_title

    # Status Helpers
    def is_confirmed(self) -> bool:
        """Check if this answer is confirmed"""
        return self.status == "confirmed"

    def confirm(self) -> None:
        """Mark this answer as confirmed"""
        self.status = "confirmed"

    def mark_confirmed(self) -> None:
        """Mark this answer as confirmed (alias for confirm())"""
        self.status = "confirmed"

    def mark_reviewed(self) -> None:
        """Mark this answer as reviewed"""
        self.status = "reviewed"

    def mark_ranked(self) -> None:
        """Mark this answer as ranked"""
        self.status = "ranked"


# Backward compatibility alias
StepAnswer = RequirementAnswer


@dataclass
class ParsedProblem:
    """
    Represents a problem file parsed into structured components.

    A problem contains REQUIREMENTS that must be answered.
    Each (1), (2), etc. in the problem becomes a Requirement.

    Supports multiple formats:
    - Flat parenthesized: (1), (2), ...
    - Flat dot-number: 1., 2., ...
    - Hierarchical: (A) + sub-steps 1., 2.
    """
    title: str                                  # Problem title
    background: str                             # Background/hypothesis section
    input_data_description: Optional[str] = None
    requirements: List[Requirement] = field(default_factory=list)
    problem_type: str = "flat"                  # "flat" or "hierarchical"
    major_sections: Dict[str, str] = field(default_factory=dict)
    format_detected: str = "flat_paren_num"

    @property
    def research_steps(self) -> List[Requirement]:
        """Backward compatibility: returns requirements as research_steps"""
        return self.requirements

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "background": self.background,
            "input_data_description": self.input_data_description,
            "requirements": [req.to_dict() for req in self.requirements],
            "research_steps": [req.to_dict() for req in self.requirements],
            "problem_type": self.problem_type,
            "major_sections": self.major_sections,
            "format_detected": self.format_detected
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedProblem":
        req_data = data.get("requirements", data.get("research_steps", []))
        reqs = [Requirement.from_dict(r) for r in req_data]
        return cls(
            title=data.get("title", ""),
            background=data.get("background", ""),
            input_data_description=data.get("input_data_description"),
            requirements=reqs,
            problem_type=data.get("problem_type", "flat"),
            major_sections=data.get("major_sections", {}),
            format_detected=data.get("format_detected", "flat_paren_num")
        )

    def get_requirement_by_id(self, requirement_id: str) -> Optional[Requirement]:
        """Get a requirement by its ID."""
        for req in self.requirements:
            if req.requirement_id == requirement_id:
                return req
        return None

    def get_step_by_id(self, step_id: str) -> Optional[Requirement]:
        """Backward compatibility: Get a requirement by its ID."""
        return self.get_requirement_by_id(step_id)

    def get_requirements_by_parent(self, parent_id: Optional[str] = None) -> List[Requirement]:
        """Get all requirements with a specific parent (None for top-level)."""
        return [r for r in self.requirements if r.parent_id == parent_id]

    def get_steps_by_parent(self, parent_id: Optional[str] = None) -> List[Requirement]:
        """Backward compatibility: Get all requirements with a specific parent."""
        return self.get_requirements_by_parent(parent_id)

    def get_execution_order(self) -> List[List[str]]:
        """
        Returns requirements grouped by execution order (parallelizable groups).
        Uses topological sort based on depends_on.
        """
        executed = set()
        result = []
        remaining = {r.requirement_id: r for r in self.requirements}

        while remaining:
            ready = []
            for req_id, req in remaining.items():
                if all(dep in executed for dep in req.depends_on):
                    ready.append(req_id)

            if not ready:
                ready = list(remaining.keys())

            result.append(ready)
            for req_id in ready:
                executed.add(req_id)
                del remaining[req_id]

        return result


@dataclass
class ResearchGoal:
    """Represents the overarching research goal"""
    description: str = ""
    problem_type: Optional[str] = None
    domain: str = "biology"
    goal_id: str = ""
    focus_areas: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.goal_id:
            self.goal_id = f"goal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass
class Paper:
    """Represents a scientific paper reference"""
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
