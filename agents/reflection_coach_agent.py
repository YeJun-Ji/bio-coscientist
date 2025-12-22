"""
Reflection Coach Agent - v6.0 Qualitative Feedback Pipeline

This agent reviews RequirementAnswers and provides qualitative feedback on 4 criteria.
NO scores - focuses on actionable feedback for user review and report generation.

Key Principles:
- Reviews answer quality using 4 evaluation criteria
- Provides SPECIFIC, ACTIONABLE feedback (coach-style)
- Does NOT score or regenerate answers
- Feedback is used for user review and report warnings
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core import RequirementAnswer
from ..core.evaluation_results import ReflectionResult, FeedbackItem
from ..external_apis import LLMClient
from ..memory import ContextMemory
from ..prompts import PromptManager
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReflectionCoachAgent(BaseAgent):
    """
    Reviews RequirementAnswers and provides qualitative feedback on 4 criteria.

    Evaluation Criteria:
    1. Logical Flow - Is the reasoning internally consistent?
    2. Requirement Coverage - Are all parts of the requirement addressed?
    3. Tool Appropriateness - Are the tools used appropriate for the task?
    4. Experimental Feasibility - Can the proposed approach be validated experimentally?

    Architecture Note:
    - NO scores (removed in v6.0)
    - Feedback stored in answer.metadata["reflection"]
    - Used for report generation (warnings/notes for confirmed answers)
    """

    CRITERIA = [
        "logical_flow",
        "requirement_coverage",
        "tool_appropriateness",
        "experimental_feasibility"
    ]

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
        self.log("ReflectionCoachAgent initialized (v6.0 - qualitative feedback, no scores)")

    # ========================================================================
    # Main Entry Point (Required by BaseAgent)
    # ========================================================================

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to reflect_on_answer.
        """
        answer = task.get("answer")
        requirement = task.get("requirement", {})

        if answer is None:
            return {"status": "error", "message": "No answer provided"}

        result = await self.reflect_on_answer(answer, requirement)
        return {
            "status": "success",
            "reflection": result.to_dict()
        }

    # ========================================================================
    # Reflection Logic
    # ========================================================================

    async def reflect_on_answer(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any]
    ) -> ReflectionResult:
        """
        Evaluate answer and provide qualitative feedback on 4 criteria.

        Args:
            answer: RequirementAnswer to evaluate
            requirement: Requirement specification

        Returns:
            ReflectionResult with feedback_items for 4 criteria

        Process:
            1. Build prompt with answer and requirement
            2. Call LLM for 4-criteria evaluation
            3. Parse feedback items
            4. Return ReflectionResult
        """
        answer_id = self._get_answer_id(answer)
        self.log(f"Reflecting on answer: {answer_id}")

        # Evaluate using LLM
        feedback_items = await self._assess_with_4_criteria(answer, requirement)

        # Log summary
        assessments = {f.criterion: f.assessment for f in feedback_items}
        weak_count = len([f for f in feedback_items if f.assessment in ("weak", "missing")])
        self.log(
            f"  Assessments: {assessments}, "
            f"Weak/Missing: {weak_count}"
        )

        return ReflectionResult(feedback_items=feedback_items)

    async def _assess_with_4_criteria(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any]
    ) -> List[FeedbackItem]:
        """
        Assess answer using LLM with 4 evaluation criteria.

        Returns:
            List of FeedbackItem for each criterion
        """
        # Build prompt
        prompt = self.prompt_manager.get_prompt(
            "reflection/qualitative_feedback",
            requirement=requirement,
            answer=self._answer_to_dict(answer)
        )

        # Call LLM
        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistency
                purpose="reflection_qualitative_feedback"
            )

            # Parse feedback items
            feedback_items = []
            raw_items = response.get("feedback_items", [])

            for item in raw_items:
                criterion = item.get("criterion", "unknown")
                if criterion not in self.CRITERIA:
                    self.log(f"  Warning: Unknown criterion '{criterion}', skipping")
                    continue

                feedback_items.append(FeedbackItem(
                    criterion=criterion,
                    assessment=item.get("assessment", "weak"),
                    observation=item.get("observation", "No observation provided"),
                    suggestion=item.get("suggestion", "No suggestion provided"),
                    evidence=item.get("evidence")
                ))

            # Ensure all 4 criteria are present
            present_criteria = {f.criterion for f in feedback_items}
            for criterion in self.CRITERIA:
                if criterion not in present_criteria:
                    self.log(f"  Warning: Missing criterion '{criterion}', adding default")
                    feedback_items.append(FeedbackItem(
                        criterion=criterion,
                        assessment="weak",
                        observation="Criterion not evaluated due to parsing error",
                        suggestion="Manual review recommended"
                    ))

            return feedback_items

        except Exception as e:
            self.log(f"ERROR in LLM quality assessment: {e}", level="error")
            # Return fallback feedback
            return [
                FeedbackItem(
                    criterion=criterion,
                    assessment="weak",
                    observation=f"Evaluation failed: {str(e)}",
                    suggestion="Manual review required due to LLM error"
                )
                for criterion in self.CRITERIA
            ]

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_answer_id(self, answer) -> str:
        """Get answer ID supporting both dict and object forms."""
        if isinstance(answer, dict):
            return answer.get("id", "unknown")
        return getattr(answer, "id", "unknown")

    def _answer_to_dict(self, answer) -> Dict[str, Any]:
        """Convert answer to dict for prompt template."""
        if isinstance(answer, dict):
            return answer

        return {
            "id": getattr(answer, "id", "unknown"),
            "answer": getattr(answer, "answer", ""),
            "rationale": getattr(answer, "rationale", ""),
            "deliverables": getattr(answer, "deliverables", {}),
            "metadata": getattr(answer, "metadata", {})
        }
