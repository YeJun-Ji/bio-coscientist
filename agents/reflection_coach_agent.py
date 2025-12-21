"""
Reflection Coach Agent - v5.0 Simplified Evaluation Pipeline

This agent reviews RequirementAnswers and provides actionable feedback.
NO feedback loop - single pass evaluation only.

Key Principles:
- Reviews answer quality objectively
- Provides SPECIFIC, ACTIONABLE feedback (coach-style)
- Scores answers but does NOT regenerate them
- All answers proceed to Tournament with their reflection scores
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core import RequirementAnswer
from ..core.evaluation_results import (
    ReflectionResult,
    ActionableFeedback,
    QualityMetrics,
    Violation
)
from ..external_apis import LLMClient
from ..memory import ContextMemory
from ..prompts import PromptManager
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReflectionCoachAgent(BaseAgent):
    """
    Reviews RequirementAnswers and provides actionable feedback.

    Architecture Note:
    - Reuses LogVerificationAgent.verify() for objective checks
    - Adds LLM-based quality assessment
    - Formula: overall_score = verification_score * 0.4 + quality_score * 0.6
    - NO regeneration loop (simplified from original design)
    """

    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        **kwargs
    ):
        super().__init__(
            name="reflection_coach",
            memory=memory,
            config=config,
            llm_client=llm_client,
            **kwargs
        )
        self.prompt_manager = prompt_manager or PromptManager()
        self.log("ReflectionCoachAgent initialized (v5.0 - simplified, no feedback loop)")

    # ========================================================================
    # Main Entry Point (Required by BaseAgent)
    # ========================================================================

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to reflect_on_answer.
        """
        return await self.reflect_on_answer(task)

    # ========================================================================
    # Reflection Logic
    # ========================================================================

    async def reflect_on_answer(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any],
        verification_results: Dict[str, Any],
        config: Dict[str, Any]
    ) -> ReflectionResult:
        """
        Evaluate answer and provide actionable feedback.

        Args:
            answer: RequirementAnswer to evaluate
            requirement: Requirement specification
            verification_results: Results from LogVerificationAgent.verify()
            config: Research configuration

        Returns:
            ReflectionResult with overall_score and actionable_feedback

        Process:
            1. Extract verification_score from LogVerificationAgent
            2. Assess quality using LLM (logical consistency, clarity, actionability)
            3. Check constraint satisfaction
            4. Generate actionable feedback for any issues
            5. Calculate overall_score = verification*0.4 + quality*0.6
        """
        self.log(f"Reflecting on answer: {answer.id}")

        # Step 1: Extract verification score
        verification_score = verification_results.get("verification_score", 0.0)
        self.log(f"  Verification score: {verification_score:.2f}")

        # Step 2: Assess quality using LLM
        quality_result = await self._assess_quality_with_llm(
            answer=answer,
            requirement=requirement,
            verification_results=verification_results
        )

        quality_score = quality_result["quality_score"]
        quality_metrics = QualityMetrics(
            evidence_alignment=quality_result["metrics"]["evidence_alignment"],
            constraint_satisfaction=quality_result["metrics"]["constraint_satisfaction"],
            logical_completeness=quality_result["metrics"]["logical_completeness"]
        )

        # Step 3: Parse actionable feedback from LLM response
        actionable_feedback = [
            ActionableFeedback(
                issue_type=fb["issue_type"],
                location=fb["location"],
                problem=fb["problem"],
                fix_instruction=fb["fix_instruction"],
                priority=fb["priority"]
            )
            for fb in quality_result.get("actionable_feedback", [])
        ]

        # Step 4: Parse violations
        violations = [
            Violation(
                type=v["type"],
                severity=v["severity"],
                description=v["description"],
                evidence=v.get("evidence")
            )
            for v in quality_result.get("violations", [])
        ]

        # Step 5: Calculate overall score
        overall_score = verification_score * 0.4 + quality_score * 0.6

        self.log(
            f"  Quality score: {quality_score:.2f}, "
            f"Overall score: {overall_score:.2f}, "
            f"Feedback items: {len(actionable_feedback)}"
        )

        return ReflectionResult(
            overall_score=overall_score,
            actionable_feedback=actionable_feedback,
            violations=violations,
            quality_metrics=quality_metrics,
            verification_score=verification_score
        )

    async def _assess_quality_with_llm(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any],
        verification_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess answer quality using LLM.

        Returns:
            {
                "quality_score": float (0.0-1.0),
                "metrics": {
                    "evidence_alignment": float,
                    "constraint_satisfaction": float,
                    "logical_completeness": float
                },
                "actionable_feedback": List[Dict],
                "violations": List[Dict]
            }
        """
        # Build prompt
        prompt = self.prompt_manager.get_prompt(
            "reflection/coach_style_reflection",
            requirement=requirement,
            answer={
                "id": answer.id,
                "answer": answer.answer,
                "rationale": answer.rationale,
                "deliverables": answer.deliverables
            },
            verification_results=verification_results
        )

        # Call LLM
        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistency
                purpose="reflection_quality_assessment"
            )

            # Calculate quality_score from metrics
            metrics = response.get("quality_metrics", {})
            quality_score = (
                metrics.get("evidence_alignment", 0.5) * 0.4 +
                metrics.get("constraint_satisfaction", 0.5) * 0.3 +
                metrics.get("logical_completeness", 0.5) * 0.3
            )

            response["quality_score"] = quality_score
            response["metrics"] = metrics

            return response

        except Exception as e:
            self.log(f"ERROR in LLM quality assessment: {e}", level="error")
            # Return fallback scores
            return {
                "quality_score": 0.5,
                "metrics": {
                    "evidence_alignment": 0.5,
                    "constraint_satisfaction": 0.5,
                    "logical_completeness": 0.5
                },
                "actionable_feedback": [
                    {
                        "issue_type": "error",
                        "location": "quality_assessment",
                        "problem": f"LLM assessment failed: {str(e)}",
                        "fix_instruction": "Review manually",
                        "priority": "critical"
                    }
                ],
                "violations": []
            }
