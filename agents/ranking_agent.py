"""
Ranking Agent - Score-Based Answer Ranking (v4.0)

⚠️ DEPRECATED in v5.0: Use TournamentRankingAgent instead.
⚠️ This agent will be removed in v6.0.

This agent ranks RequirementAnswers using a simple composite score.
Replaces the complex ELO tournament system with transparent score-based sorting.

Key Features:
- Simple composite score: verification_score * 0.5 + quality_score * 0.5
- Instant ranking (no LLM calls, <0.01s)
- Transparent and deterministic
- Updates ELO ratings for backward compatibility

Ranking Formula:
    composite_score = verification_score * 0.5 + quality_score * 0.5

Where:
- verification_score: Objective (from LogVerificationAgent, fact-checked against logs)
- quality_score: Subjective (from QualityAssessmentAgent, domain-agnostic evaluation)
- 50/50 weight: Balances objective facts and subjective quality
"""

import logging
import warnings
from typing import Dict, Any, List

from ..core import RequirementAnswer
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """
    Ranking agent - sorts answers by composite score.

    Unlike old ELO tournament (3-4 LLM calls, pairwise comparisons),
    this agent uses simple score-based sorting:
    1. Calculate composite_score for each answer
    2. Sort by composite_score (descending)
    3. Update ELO ratings for convergence check compatibility

    NO LLM calls - instant ranking.
    """

    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(
            "ranking",
            memory,
            config,
            llm_client=None,  # No LLM needed!
            **kwargs
        )

        # Configurable weights (default: 50/50)
        self.verification_weight = config.get("ranking", {}).get("verification_weight", 0.5)
        self.quality_weight = config.get("ranking", {}).get("quality_weight", 0.5)

        # Validate weights sum to 1.0
        total_weight = self.verification_weight + self.quality_weight
        if abs(total_weight - 1.0) > 0.001:
            self.log(
                f"⚠ Ranking weights don't sum to 1.0 ({total_weight}), normalizing...",
                "warning"
            )
            self.verification_weight /= total_weight
            self.quality_weight /= total_weight

    # ========== RequirementAnswer Helper Methods ==========

    def _get_answer_id(self, answer) -> str:
        """Get RequirementAnswer ID from dict or object"""
        if isinstance(answer, dict):
            return answer.get("id", "unknown")
        return getattr(answer, "id", "unknown")

    def _get_verification_score(self, answer) -> float:
        """Get verification_score from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            # Try new field first, fall back to metadata
            if "verification_score" in answer:
                return float(answer["verification_score"])
            return float(answer.get("metadata", {}).get("verification_score", 0.0))

        # Object form
        if hasattr(answer, "verification_score"):
            return float(answer.verification_score)
        return float(getattr(answer, "metadata", {}).get("verification_score", 0.0))

    def _get_quality_score(self, answer) -> float:
        """Get quality_score from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return float(answer.get("quality_score", 0.0))
        return float(getattr(answer, "quality_score", 0.0))

    def _set_composite_score(self, answer, score: float):
        """Set composite_score on dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            answer["composite_score"] = score
        else:
            answer.composite_score = score

    def _set_elo_rating(self, answer, rating: float):
        """Set ELO rating on dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            answer["elo_rating"] = rating
        else:
            answer.elo_rating = rating

    # ========== Main Entry Point ==========

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Ranks answers by composite score.

        Args:
            task: {
                "answers": List[RequirementAnswer] (after quality assessment)
            }

        Returns:
            {
                "status": "success" | "error",
                "ranked_answers": List[RequirementAnswer] (sorted by composite_score),
                "scores": List[Dict] (ranking details)
            }
        """
        # DEPRECATED v5.0: Warn users to migrate to TournamentRankingAgent
        warnings.warn(
            "RankingAgent (score-based) is deprecated in v5.0. "
            "Use TournamentRankingAgent for relative evaluation instead. "
            "This agent will be removed in v6.0.",
            DeprecationWarning,
            stacklevel=2
        )

        answers = task.get("answers", [])

        if not answers:
            self.log("WARNING: No answers provided for ranking", "warning")
            return {
                "status": "success",
                "ranked_answers": [],
                "scores": []
            }

        self.log(f"[RANKING] Ranking {len(answers)} answers by composite score")

        try:
            ranked_answers = self.rank(answers)

            # Extract scores for logging
            scores = [
                {
                    "answer_id": self._get_answer_id(ans),
                    "composite_score": getattr(ans, "composite_score", 0.0)
                        if not isinstance(ans, dict) else ans.get("composite_score", 0.0),
                    "verification_score": self._get_verification_score(ans),
                    "quality_score": self._get_quality_score(ans),
                    "elo_rating": getattr(ans, "elo_rating", 1200.0)
                        if not isinstance(ans, dict) else ans.get("elo_rating", 1200.0)
                }
                for ans in ranked_answers
            ]

            self.log(f"[RANKING] ✓ Ranking complete")
            for i, score_info in enumerate(scores, 1):
                self.log(
                    f"[RANKING]   {i}. {score_info['answer_id']}: "
                    f"composite={score_info['composite_score']:.3f} "
                    f"(verify={score_info['verification_score']:.3f}, "
                    f"quality={score_info['quality_score']:.3f})"
                )

            return {
                "status": "success",
                "ranked_answers": ranked_answers,
                "scores": scores
            }

        except Exception as e:
            self.log(f"[RANKING] ✗ Error: {e}", "error")
            import traceback
            self.log(f"[RANKING] {traceback.format_exc()}", "debug")
            return {
                "status": "error",
                "message": str(e)
            }

    # ========== Core Ranking Method ==========

    def rank(self, answers: List) -> List:
        """
        Rank answers by composite score.

        This is the core method - simple and transparent.

        Process:
        1. Calculate composite_score for each answer
        2. Sort by composite_score (descending)
        3. Update ELO ratings for backward compatibility

        Args:
            answers: List of RequirementAnswer (dict or object)

        Returns:
            Sorted list (highest composite_score first)
        """
        import time
        start_time = time.time()

        self.log(f"  │ Computing composite scores...")

        # Calculate composite scores
        for answer in answers:
            verification_score = self._get_verification_score(answer)
            quality_score = self._get_quality_score(answer)

            # Composite score: weighted average
            composite_score = (
                verification_score * self.verification_weight +
                quality_score * self.quality_weight
            )

            # Ensure valid range
            composite_score = max(0.0, min(1.0, composite_score))

            # Set composite score
            self._set_composite_score(answer, composite_score)

        # Sort by composite_score (descending)
        self.log(f"  │ Sorting by composite_score...")
        ranked_answers = sorted(
            answers,
            key=lambda ans: getattr(ans, "composite_score", 0.0)
                if not isinstance(ans, dict) else ans.get("composite_score", 0.0),
            reverse=True
        )

        # Update ELO ratings for backward compatibility
        # (Supervisor's convergence check uses answer.elo_rating)
        self.log(f"  │ Updating ELO ratings for backward compatibility...")
        base_elo = 1200.0
        elo_increment = 50.0

        for i, answer in enumerate(ranked_answers):
            # Higher rank = higher ELO
            # Rank 1: 1200 + (n-0)*50 = 1200 + n*50
            # Rank 2: 1200 + (n-1)*50
            # ...
            # Rank n: 1200 + 50
            rank = i
            elo_rating = base_elo + (len(ranked_answers) - rank) * elo_increment
            self._set_elo_rating(answer, elo_rating)

        duration = time.time() - start_time
        self.log(f"  │ ✓ Ranking complete in {duration:.4f}s")

        return ranked_answers
