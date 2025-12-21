"""
Evaluation Result Data Structures for v5.0 Three-Agent Pipeline

This module defines data structures for:
- Reflection Agent: ActionableFeedback, ReflectionResult
- Tournament Ranking Agent: Matchup, TournamentResult
- Evolution Agent: ProtocolModule, LiteratureModule, RiskModule, EnrichmentResult
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================================
# Reflection Agent Data Structures
# ============================================================================

@dataclass
class ActionableFeedback:
    """
    Actionable feedback from Reflection Agent (Coach-style).

    Example:
        BAD: "Constraint violation."
        GOOD: "Config requires residues 50-60, but answer includes residue 70.
               Fix: Re-run docking constrained to 50-60 range."
    """
    issue_type: str  # "constraint_violation", "evidence_gap", "logical_error", "improvement"
    location: str  # "binding site section", "BLAST analysis", etc.
    problem: str  # What's wrong or could be improved
    fix_instruction: str  # How to fix (CRITICAL - must be specific)
    priority: str  # "critical", "important", "minor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "location": self.location,
            "problem": self.problem,
            "fix_instruction": self.fix_instruction,
            "priority": self.priority
        }


@dataclass
class QualityMetrics:
    """Quality assessment metrics from Reflection Agent."""
    evidence_alignment: float  # 0.0-1.0
    constraint_satisfaction: float  # 0.0-1.0
    logical_completeness: float  # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_alignment": self.evidence_alignment,
            "constraint_satisfaction": self.constraint_satisfaction,
            "logical_completeness": self.logical_completeness
        }


@dataclass
class Violation:
    """Detailed violation information."""
    type: str  # "hallucination", "constraint_violation", "logic_gap", "missing_deliverable"
    severity: str  # "critical", "major", "minor"
    description: str
    evidence: Optional[str] = None  # Supporting evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence
        }


@dataclass
class ReflectionResult:
    """
    Result from Reflection Agent evaluation.

    NO "status" or "iteration" fields since there's no feedback loop.
    """
    overall_score: float  # verification*0.4 + quality*0.6 (0.0-1.0)
    actionable_feedback: List[ActionableFeedback]
    violations: List[Violation]
    quality_metrics: QualityMetrics
    verification_score: float  # From LogVerificationAgent
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "actionable_feedback": [f.to_dict() for f in self.actionable_feedback],
            "violations": [v.to_dict() for v in self.violations],
            "quality_metrics": self.quality_metrics.to_dict(),
            "verification_score": self.verification_score,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# Tournament Ranking Agent Data Structures
# ============================================================================

@dataclass
class Matchup:
    """Single pairwise comparison in tournament."""
    answer_a_id: str
    answer_b_id: str
    winner_id: str
    reasoning: str  # LLM's detailed comparison reasoning
    criteria_scores: Dict[str, Dict[str, float]]  # {goal_contribution: {a: 0.8, b: 0.6}, ...}
    margin: str  # "clear", "close", "very_close"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_a_id": self.answer_a_id,
            "answer_b_id": self.answer_b_id,
            "winner_id": self.winner_id,
            "reasoning": self.reasoning,
            "criteria_scores": self.criteria_scores,
            "margin": self.margin
        }


@dataclass
class TournamentResult:
    """Result from tournament-based ranking."""
    final_rank: int  # 1, 2, 3...
    elo_rating: float  # Updated ELO (backward compatible)
    matchups: List[Matchup]  # All matchups this answer participated in
    wins: int
    losses: int
    strengths: List[str]  # Why this answer ranked well
    weaknesses: List[str]  # Areas for improvement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_rank": self.final_rank,
            "elo_rating": self.elo_rating,
            "matchups": [m.to_dict() for m in self.matchups],
            "wins": self.wins,
            "losses": self.losses,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses
        }


# ============================================================================
# Evolution Agent Data Structures
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
