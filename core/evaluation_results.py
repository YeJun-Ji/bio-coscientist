"""
Evaluation Result Data Structures for v6.0 Phase Pipeline

This module defines data structures for:
- Reflection Agent: FeedbackItem, ReflectionResult (4-criteria qualitative feedback)
- Tournament Ranking Agent: Matchup, TournamentResult (win-count based)
- Evolution Agent: ProtocolModule, LiteratureModule, RiskModule, EnrichmentResult
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================================
# Reflection Agent Data Structures (v6.0 - Qualitative Feedback)
# ============================================================================

@dataclass
class FeedbackItem:
    """
    Single feedback item for one of the 4 evaluation criteria.

    Criteria:
    - logical_flow: Is the reasoning internally consistent?
    - requirement_coverage: Are all parts of the requirement addressed?
    - tool_appropriateness: Are the tools used appropriate for the task?
    - experimental_feasibility: Can the proposed approach be validated experimentally?
    """
    criterion: str  # "logical_flow", "requirement_coverage", "tool_appropriateness", "experimental_feasibility"
    assessment: str  # "strong", "adequate", "weak", "missing"
    observation: str  # What was observed in the answer
    suggestion: str  # How to improve (even if strong)
    evidence: Optional[str] = None  # Quote from answer (optional)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "assessment": self.assessment,
            "observation": self.observation,
            "suggestion": self.suggestion,
            "evidence": self.evidence
        }


@dataclass
class ReflectionResult:
    """
    Result from Reflection Agent evaluation (v6.0).

    Provides qualitative feedback on 4 criteria without numerical scores.
    Used for user feedback and report generation (as warnings/notes).
    """
    feedback_items: List[FeedbackItem]  # 4 criteria feedback
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_items": [f.to_dict() for f in self.feedback_items],
            "timestamp": self.timestamp.isoformat()
        }

    def get_criterion(self, name: str) -> Optional[FeedbackItem]:
        """Get feedback for a specific criterion."""
        for item in self.feedback_items:
            if item.criterion == name:
                return item
        return None

    def get_weak_criteria(self) -> List[FeedbackItem]:
        """Get all criteria with weak or missing assessment."""
        return [f for f in self.feedback_items if f.assessment in ("weak", "missing")]

    def has_critical_issues(self) -> bool:
        """Check if any criterion has missing assessment."""
        return any(f.assessment == "missing" for f in self.feedback_items)


# ============================================================================
# Tournament Ranking Agent Data Structures (v6.0 - Win-count based)
# ============================================================================

@dataclass
class Matchup:
    """
    Single pairwise comparison in tournament (v6.0 simplified).

    No criteria_scores - just winner decision with reasoning.
    """
    answer_a_id: str
    answer_b_id: str
    winner_id: str
    reasoning: str  # LLM's comparison reasoning
    margin: str  # "clear", "close", "very_close"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_a_id": self.answer_a_id,
            "answer_b_id": self.answer_b_id,
            "winner_id": self.winner_id,
            "reasoning": self.reasoning,
            "margin": self.margin
        }


@dataclass
class TournamentResult:
    """
    Result from tournament-based ranking (v6.0).

    Uses pure win-count for ranking, no ELO.
    """
    final_rank: int  # 1, 2, 3...
    matchups: List[Matchup]  # All matchups this answer participated in
    wins: int
    losses: int
    is_novelty_winner: bool = False  # True if won via novelty tiebreaker

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_rank": self.final_rank,
            "matchups": [m.to_dict() for m in self.matchups],
            "wins": self.wins,
            "losses": self.losses,
            "is_novelty_winner": self.is_novelty_winner
        }


# ============================================================================
# Evolution Agent Data Structures (Retained for future use)
# ============================================================================

@dataclass
class ExperimentalStep:
    """Single step in experimental protocol."""
    step_number: int
    action: str
    duration: str
    materials: List[str]
    expected_result: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "action": self.action,
            "duration": self.duration,
            "materials": self.materials,
            "expected_result": self.expected_result
        }


@dataclass
class ProtocolModule:
    """Module 1: Experimental validation protocol."""
    steps: List[ExperimentalStep]
    estimated_cost: Optional[str] = None
    estimated_duration: Optional[str] = None
    required_equipment: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "estimated_cost": self.estimated_cost,
            "estimated_duration": self.estimated_duration,
            "required_equipment": self.required_equipment
        }


@dataclass
class Paper:
    """Scientific paper reference."""
    title: str
    pmid: str
    relevance_score: float
    key_finding: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "pmid": self.pmid,
            "relevance_score": self.relevance_score,
            "key_finding": self.key_finding
        }


@dataclass
class LiteratureModule:
    """Module 2: Literature support via PubMed."""
    supporting_papers: List[Paper]
    contradicting_papers: List[Paper]
    evidence_strength: str  # "strong", "moderate", "weak"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supporting_papers": [p.to_dict() for p in self.supporting_papers],
            "contradicting_papers": [p.to_dict() for p in self.contradicting_papers],
            "evidence_strength": self.evidence_strength
        }


@dataclass
class Risk:
    """Identified risk."""
    type: str  # "technical", "biological", "validation"
    description: str
    likelihood: str  # "low", "medium", "high"
    impact: str  # "low", "medium", "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "likelihood": self.likelihood,
            "impact": self.impact
        }


@dataclass
class MitigationStrategy:
    """Risk mitigation strategy."""
    for_risk: str  # Which risk this addresses
    strategy: str
    effectiveness: str  # "80% success rate", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "for_risk": self.for_risk,
            "strategy": self.strategy,
            "effectiveness": self.effectiveness
        }


@dataclass
class RiskModule:
    """Module 3: Risk analysis and mitigation."""
    risks: List[Risk]
    mitigation_strategies: List[MitigationStrategy]
    overall_risk_level: str  # "low", "medium", "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risks": [r.to_dict() for r in self.risks],
            "mitigation_strategies": [m.to_dict() for m in self.mitigation_strategies],
            "overall_risk_level": self.overall_risk_level
        }


@dataclass
class EnrichmentResult:
    """
    Result from Evolution Agent's three parallel modules.

    IMPORTANT: Only applied to the SINGLE CONFIRMED ANSWER (tournament winner).
    Currently returns None (v6.0 - disabled, for future use).
    """
    protocol: Optional[ProtocolModule]
    literature: Optional[LiteratureModule]
    risk: Optional[RiskModule]
    overall_confidence: float  # Composite confidence from all modules
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict() if self.protocol else None,
            "literature": self.literature.to_dict() if self.literature else None,
            "risk": self.risk.to_dict() if self.risk else None,
            "overall_confidence": self.overall_confidence,
            "timestamp": self.timestamp.isoformat()
        }
