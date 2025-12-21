"""
Quality Assessment Agent - Domain-Agnostic Quality Evaluation (v4.0)

⚠️ DEPRECATED in v5.0: Use ReflectionCoachAgent instead.
⚠️ This agent will be removed in v6.0.

This agent evaluates RequirementAnswer quality using domain-agnostic criteria.
Works for ALL biomedical domains without code changes.

Key Features:
- Single LLM call (fast, ~3-5s)
- Domain-agnostic criteria (logic, clarity, actionability)
- Uses verification results as context
- NO tool-specific thresholds (verification already checked)

Evaluation Dimensions:
1. Logical Consistency (40%): Evidence → Reasoning → Conclusion flow
2. Clarity (30%): Clear structure and communication
3. Actionability (30%): Concrete deliverables and next steps
"""

import logging
import warnings
from typing import Dict, Any, Optional

from ..core import RequirementAnswer
from ..external_apis import LLMClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class QualityAssessmentAgent(BaseAgent):
    """
    Quality assessment agent - evaluates answer quality using domain-agnostic criteria.

    Unlike old AbsoluteEvaluationAgent with biology-specific thresholds,
    this agent works for ANY domain by focusing on universal quality criteria.

    Single LLM call evaluates:
    - Logical Consistency (40%): Is reasoning sound?
    - Clarity (30%): Is it understandable?
    - Actionability (30%): Are deliverables concrete?

    Output: quality_score (0.0 to 1.0) with dimensional feedback
    """

    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        **kwargs
    ):
        super().__init__(
            "quality_assessment",
            memory,
            config,
            llm_client,
            **kwargs
        )

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

    def _get_answer_metadata(self, answer) -> Dict[str, Any]:
        """Get metadata from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("metadata", {})
        return getattr(answer, "metadata", {})

    def _format_answer_for_template(self, answer) -> Dict[str, Any]:
        """Format answer for template rendering"""
        return {
            "id": self._get_answer_id(answer),
            "answer": self._get_answer_content(answer),
            "rationale": self._get_answer_rationale(answer),
            "deliverables": self._get_answer_deliverables(answer),
            "metadata": self._get_answer_metadata(answer)
        }

    def _build_context_dict(self, context: Dict[str, Any]) -> Dict[str, Dict]:
        """Build context dict for template rendering"""
        if not context:
            return {}

        context_dict = {}
        for dep_id, dep_answer in context.items():
            if isinstance(dep_answer, dict):
                context_dict[dep_id] = {
                    "requirement_title": dep_answer.get("requirement_title", ""),
                    "answer": dep_answer.get("answer", "")
                }
            else:
                context_dict[dep_id] = {
                    "requirement_title": getattr(dep_answer, "requirement_title", ""),
                    "answer": getattr(dep_answer, "answer", "")
                }
        return context_dict

    # ========== Main Entry Point ==========

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Evaluates answer quality using domain-agnostic criteria.

        Args:
            task: {
                "answer": RequirementAnswer (after verification),
                "requirement": Dict (requirement specification),
                "context": Dict[str, RequirementAnswer] (confirmed dependency answers),
                "verification_results": Dict (from LogVerificationAgent, optional)
            }

        Returns:
            {
                "status": "success" | "error",
                "answer_id": str,
                "quality_score": float (0.0 to 1.0),
                "dimensions": {
                    "logical_consistency": {"score": float, "feedback": str},
                    "clarity": {"score": float, "feedback": str},
                    "actionability": {"score": float, "feedback": str}
                },
                "overall_feedback": str,
                "strengths": List[str],
                "weaknesses": List[str]
            }
        """
        # DEPRECATED v5.0: Warn users to migrate to ReflectionCoachAgent
        warnings.warn(
            "QualityAssessmentAgent is deprecated in v5.0. "
            "Use ReflectionCoachAgent for actionable feedback instead. "
            "This agent will be removed in v6.0.",
            DeprecationWarning,
            stacklevel=2
        )

        answer = task.get("answer")
        requirement = task.get("requirement", {})
        context = task.get("context", {})
        verification_results = task.get("verification_results", None)

        if not answer:
            self.log("ERROR: 'answer' not provided in task", "error")
            return {"status": "error", "message": "answer not provided"}

        answer_id = self._get_answer_id(answer)
        req_id = self._get_answer_requirement_id(answer)

        self.log(f"[QUALITY-ASSESSMENT] Evaluating {answer_id} for requirement {req_id}")

        try:
            result = await self.assess(answer, requirement, context, verification_results)
            self.log(f"[QUALITY-ASSESSMENT] ✓ Complete - Quality Score: {result['quality_score']:.2f}")
            return result

        except Exception as e:
            self.log(f"[QUALITY-ASSESSMENT] ✗ Error: {e}", "error")
            import traceback
            self.log(f"[QUALITY-ASSESSMENT] {traceback.format_exc()}", "debug")
            return {
                "status": "error",
                "answer_id": answer_id,
                "message": str(e)
            }

    # ========== Core Assessment Method ==========

    async def assess(
        self,
        answer,
        requirement: Dict,
        context: Dict[str, Any],
        verification_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Assess answer quality using domain-agnostic criteria.

        This is the core method - single LLM call with generic quality prompt.

        Args:
            answer: RequirementAnswer (dict or object)
            requirement: Requirement specification
            context: Confirmed dependency answers
            verification_results: Results from LogVerificationAgent (optional)

        Returns:
            {
                "status": "success",
                "answer_id": str,
                "quality_score": float,
                "dimensions": {...},
                "overall_feedback": str,
                "strengths": [...],
                "weaknesses": [...]
            }
        """
        import time
        start_time = time.time()

        answer_id = self._get_answer_id(answer)

        self.log(f"  │ Running domain-agnostic quality assessment...")

        try:
            # Render prompt with domain-agnostic template
            prompt = self.prompt_manager.get_prompt(
                "quality_assessment/domain_agnostic",
                requirement=requirement,
                answer=self._format_answer_for_template(answer),
                verification_results=verification_results,
                context=self._build_context_dict(context)
            )

            # Single LLM call
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Moderate - consistent but not rigid
                purpose=f"quality_assessment_{answer_id}"
            )

            # Extract scores
            quality_score = float(response.get("quality_score", 0.5))
            dimensions = response.get("dimensions", {})
            overall_feedback = response.get("overall_feedback", "")
            strengths = response.get("strengths", [])
            weaknesses = response.get("weaknesses", [])

            # Validate score calculation (should match template formula)
            logical_score = dimensions.get("logical_consistency", {}).get("score", 0.5)
            clarity_score = dimensions.get("clarity", {}).get("score", 0.5)
            actionability_score = dimensions.get("actionability", {}).get("score", 0.5)

            expected_score = (
                logical_score * 0.40 +
                clarity_score * 0.30 +
                actionability_score * 0.30
            )

            # Allow small floating-point tolerance
            if abs(quality_score - expected_score) > 0.01:
                self.log(
                    f"  │ ⚠ Score mismatch (reported: {quality_score:.2f}, "
                    f"expected: {expected_score:.2f}), using expected",
                    "warning"
                )
                quality_score = expected_score

            # Ensure valid range
            quality_score = max(0.0, min(1.0, quality_score))

            # Update answer object with quality score
            if isinstance(answer, dict):
                answer["quality_score"] = quality_score
                if "metadata" not in answer:
                    answer["metadata"] = {}
                answer["metadata"]["quality_assessment"] = {
                    "dimensions": dimensions,
                    "overall_feedback": overall_feedback,
                    "strengths": strengths,
                    "weaknesses": weaknesses
                }
            else:
                answer.quality_score = quality_score
                if not answer.metadata:
                    answer.metadata = {}
                answer.metadata["quality_assessment"] = {
                    "dimensions": dimensions,
                    "overall_feedback": overall_feedback,
                    "strengths": strengths,
                    "weaknesses": weaknesses
                }

            duration = time.time() - start_time
            self.log(f"  │ ✓ Assessment complete in {duration:.2f}s")
            self.log(f"  │   Logical: {logical_score:.2f}, Clarity: {clarity_score:.2f}, "
                    f"Actionability: {actionability_score:.2f}")

            return {
                "status": "success",
                "answer_id": answer_id,
                "quality_score": quality_score,
                "dimensions": dimensions,
                "overall_feedback": overall_feedback,
                "strengths": strengths,
                "weaknesses": weaknesses
            }

        except Exception as e:
            self.log(f"  │ ✗ Quality assessment error: {e}", "error")

            # Return fallback score
            fallback_score = 0.5
            return {
                "status": "error",
                "answer_id": answer_id,
                "quality_score": fallback_score,
                "dimensions": {
                    "logical_consistency": {"score": fallback_score, "feedback": "Error during assessment"},
                    "clarity": {"score": fallback_score, "feedback": "Error during assessment"},
                    "actionability": {"score": fallback_score, "feedback": "Error during assessment"}
                },
                "overall_feedback": f"Error during quality assessment: {e}",
                "strengths": [],
                "weaknesses": [f"Assessment failed: {e}"]
            }
