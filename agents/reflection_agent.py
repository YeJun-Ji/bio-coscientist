"""
Reflection Agent - RequirementAnswer Review System

This agent reviews and evaluates RequirementAnswer objects for quality,
completeness, and consistency with dependencies in the Sequential Confirmation workflow.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import ResearchGoal, Requirement, RequirementAnswer
from ..external_apis import LLMClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """
    Reviews and evaluates RequirementAnswers for the Sequential Confirmation workflow.

    Evaluation criteria:
    - Deliverable completeness: Are all expected deliverables provided?
    - Answer quality: Is the answer specific, concrete, and well-reasoned?
    - Consistency: Is it consistent with confirmed dependency answers?
    - Scientific validity: Is the rationale scientifically sound?
    - Novelty: Does it offer unique insights?
    """

    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        tool_registry=None,
        mcp_server_manager=None,
        **kwargs
    ):
        super().__init__(
            "reflection",
            memory,
            config,
            llm_client,
            tool_registry=tool_registry,
            mcp_server_manager=mcp_server_manager,
            **kwargs
        )
        # Web search client (optional)
        self.web_search = None

    # ========== RequirementAnswer Helper Methods ==========

    def _get_answer_id(self, answer) -> str:
        """Get RequirementAnswer ID from dict or object"""
        if isinstance(answer, dict):
            return answer.get("id", "unknown")
        return getattr(answer, "id", "unknown")

    def _get_answer_content(self, answer) -> str:
        """Get answer content from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("answer", "")
        return getattr(answer, "answer", "")

    def _get_answer_requirement_id(self, answer) -> str:
        """Get requirement_id from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("requirement_id", "")
        return getattr(answer, "requirement_id", "")

    def _get_answer_deliverables(self, answer) -> Dict[str, Any]:
        """Get deliverables from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("deliverables", {})
        return getattr(answer, "deliverables", {})

    def _get_answer_rationale(self, answer) -> str:
        """Get rationale from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("rationale", "")
        return getattr(answer, "rationale", "")

    def _get_answer_confidence(self, answer) -> float:
        """Get confidence from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return float(answer.get("confidence", 0.5))
        return getattr(answer, "confidence", 0.5)

    # ========== Main Entry Point ==========

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to run_for_answer for RequirementAnswer review.
        """
        return await self.run_for_answer(task)

    async def run_for_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review a RequirementAnswer (main entry point for Sequential Confirmation).

        Args:
            task: {
                "answer": RequirementAnswer or Dict,
                "requirement": Dict or Requirement object,
                "context": Dict[str, RequirementAnswer] - confirmed answers from dependencies
            }

        Returns:
            {
                "status": "success" | "error",
                "answer_id": str,
                "review": Dict,
                "quality_score": float,
                "novelty_score": float,
                "pass": bool
            }
        """
        import time
        func_start = time.time()

        answer = task.get("answer")
        requirement = task.get("requirement", {})
        context = task.get("context", {})

        if not answer:
            self.log("ERROR: 'answer' not provided in task", "error")
            return {"status": "error", "message": "answer not provided"}

        answer_id = self._get_answer_id(answer)
        req_id = self._get_answer_requirement_id(answer)

        self.log(f"[ANSWER-REVIEW] Reviewing answer {answer_id} for requirement {req_id}")

        try:
            # Perform answer review
            review_result = await self._review_requirement_answer(
                answer=answer,
                requirement=requirement,
                context=context
            )

            # Extract scores
            quality_score = review_result.get("quality_score", 0.5)
            novelty_score = review_result.get("novelty_score", 0.5)
            pass_review = review_result.get("pass", False)

            # Update answer object with review results
            if isinstance(answer, dict):
                answer["review"] = review_result
                answer["quality_score"] = quality_score
                answer["novelty_score"] = novelty_score
                answer["status"] = "reviewed"
            else:
                answer.review = review_result
                answer.quality_score = quality_score
                answer.novelty_score = novelty_score
                answer.mark_reviewed()  # Use helper method

            func_duration = time.time() - func_start
            self.log(f"[ANSWER-REVIEW] ✓ Complete ({func_duration:.2f}s)")
            self.log(f"[ANSWER-REVIEW] Quality: {quality_score:.2f}, Novelty: {novelty_score:.2f}, Pass: {pass_review}")

            return {
                "status": "success",
                "answer_id": answer_id,
                "review": review_result,
                "quality_score": quality_score,
                "novelty_score": novelty_score,
                "pass": pass_review
            }

        except Exception as e:
            self.log(f"[ANSWER-REVIEW] ✗ Error: {e}", "error")
            import traceback
            self.log(f"[ANSWER-REVIEW] {traceback.format_exc()}", "debug")
            return {
                "status": "error",
                "answer_id": answer_id,
                "message": str(e)
            }

    async def _review_requirement_answer(
        self,
        answer,
        requirement: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review a RequirementAnswer for quality, completeness, and consistency.

        Evaluation criteria:
        1. Deliverable completeness: Does it provide all expected deliverables?
        2. Answer quality: Is the answer specific, concrete, and well-reasoned?
        3. Consistency: Is it consistent with confirmed dependency answers?
        4. Scientific validity: Is the rationale scientifically sound?
        5. Novelty: Does it offer unique insights?

        Args:
            answer: RequirementAnswer to review
            requirement: The requirement being answered
            context: Confirmed answers from dependencies

        Returns:
            Review result dict
        """
        # Extract answer details
        answer_content = self._get_answer_content(answer)
        answer_rationale = self._get_answer_rationale(answer)
        answer_deliverables = self._get_answer_deliverables(answer)
        answer_confidence = self._get_answer_confidence(answer)

        # Extract requirement details
        if isinstance(requirement, dict):
            req_id = requirement.get("requirement_id", requirement.get("step_id", ""))
            req_title = requirement.get("title", "")
            req_description = requirement.get("description", "")
            expected_deliverables = requirement.get("expected_deliverables", [])
        else:
            req_id = getattr(requirement, "requirement_id", "")
            req_title = getattr(requirement, "title", "")
            req_description = getattr(requirement, "description", "")
            expected_deliverables = getattr(requirement, "expected_deliverables", [])

        # Build context summary
        context_summary = ""
        if context:
            context_parts = []
            for dep_id, dep_answer in context.items():
                if isinstance(dep_answer, dict):
                    dep_text = dep_answer.get("answer", "")[:200]
                else:
                    dep_text = getattr(dep_answer, "answer", "")[:200]
                context_parts.append(f"- {dep_id}: {dep_text}...")
            context_summary = "\n".join(context_parts)

        # Get parsed problem for background
        parsed_problem = self.config.get("parsed_problem", {})

        # Build review prompt
        prompt = self.prompt_manager.get_prompt(
            "reflection/answer_review",
            parsed_problem=parsed_problem,
            requirement={
                "requirement_id": req_id,
                "title": req_title,
                "description": req_description,
                "expected_deliverables": expected_deliverables
            },
            answer={
                "answer": answer_content,
                "rationale": answer_rationale,
                "deliverables": answer_deliverables,
                "confidence": answer_confidence
            },
            context_summary=context_summary
        )

        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                purpose=f"answer_review_{req_id}"
            )

            # Parse review response
            review_result = {
                "answer_id": self._get_answer_id(answer),
                "requirement_id": req_id,
                "deliverable_completeness": response.get("deliverable_completeness", {}),
                "quality_assessment": response.get("quality_assessment", {}),
                "consistency_assessment": response.get("consistency_assessment", {}),
                "scientific_validity": response.get("scientific_validity", {}),
                "novelty_assessment": response.get("novelty_assessment", {}),
                "strengths": response.get("strengths", []),
                "weaknesses": response.get("weaknesses", []),
                "suggestions": response.get("suggestions", []),
                "quality_score": float(response.get("quality_score", 0.5)),
                "novelty_score": float(response.get("novelty_score", 0.5)),
                "pass": response.get("pass", False),
                "overall_feedback": response.get("overall_feedback", "")
            }

            return review_result

        except Exception as e:
            self.log(f"Error in answer review: {e}", "error")
            # Return default review on error
            return {
                "answer_id": self._get_answer_id(answer),
                "requirement_id": req_id,
                "quality_score": 0.5,
                "novelty_score": 0.5,
                "pass": False,
                "error": str(e)
            }
