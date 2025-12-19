"""
Evolution Agent - RequirementAnswer Evolution System

This agent evolves and improves RequirementAnswers through various
strategies for the Sequential Confirmation workflow.
"""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import ResearchGoal, RequirementAnswer
from ..external_apis import LLMClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """
    Evolves and improves RequirementAnswers for the Sequential Confirmation workflow.

    Evolution strategies:
    - Grounding: Add more specific details and evidence
    - Coherence: Improve alignment with confirmed dependencies
    - Simplification: Remove unnecessary complexity
    - Divergent: Explore alternative approaches
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
            "evolution",
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

    def _get_answer_rationale(self, answer) -> str:
        """Get rationale from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("rationale", "")
        return getattr(answer, "rationale", "")

    def _get_answer_deliverables(self, answer) -> Dict[str, Any]:
        """Get deliverables from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("deliverables", {})
        return getattr(answer, "deliverables", {})

    def _get_answer_review(self, answer) -> Optional[Dict[str, Any]]:
        """Get review from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("review")
        return getattr(answer, "review", None)

    def _get_answer_iteration(self, answer) -> int:
        """Get iteration from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("iteration", 1)
        return getattr(answer, "iteration", 1)

    def _get_answer_builds_on(self, answer) -> List[str]:
        """Get builds_on from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("builds_on", [])
        return getattr(answer, "builds_on", [])

    # ========== Main Entry Point ==========

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to run_for_answer for RequirementAnswer evolution.
        """
        return await self.run_for_answer(task)

    async def run_for_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve a RequirementAnswer to create an improved version.

        This is the main entry point for Sequential Confirmation.
        Takes a parent answer and applies an evolution method to create
        an improved child answer.

        Args:
            task: {
                "answer": RequirementAnswer,           # Parent answer to evolve
                "requirement": Dict,                   # The requirement being answered
                "context": Dict[str, RequirementAnswer],  # Confirmed answers from dependencies
                "method": str                          # Evolution method (default: "grounding")
            }

        Returns:
            {
                "status": "success" | "error",
                "evolved_answer": RequirementAnswer,
                "method": str,
                "improvements": List[str]
            }
        """
        import time
        func_start = time.time()

        answer = task.get("answer")
        requirement = task.get("requirement", {})
        context = task.get("context", {})
        method = task.get("method", "grounding")

        if not answer:
            self.log("ERROR: 'answer' not provided in task", "error")
            return {"status": "error", "message": "answer not provided"}

        answer_id = self._get_answer_id(answer)
        req_id = self._get_answer_requirement_id(answer)

        self.log(f"[ANSWER-EVOLUTION] Evolving answer {answer_id} using '{method}'")

        try:
            # Apply evolution method
            if method == "grounding":
                evolved = await self._evolve_answer_grounding(answer, requirement, context)
            elif method == "coherence":
                evolved = await self._evolve_answer_coherence(answer, requirement, context)
            elif method == "simplification":
                evolved = await self._evolve_answer_simplification(answer, requirement, context)
            elif method == "divergent":
                evolved = await self._evolve_answer_divergent(answer, requirement, context)
            else:
                # Default to grounding
                evolved = await self._evolve_answer_grounding(answer, requirement, context)

            if evolved:
                # Store in memory
                self.memory.store_requirement_answer(evolved)

                func_duration = time.time() - func_start
                self.log(f"[ANSWER-EVOLUTION] ✓ Created evolved answer {evolved.id} ({func_duration:.2f}s)")

                return {
                    "status": "success",
                    "evolved_answer": evolved,
                    "method": method,
                    "improvements": evolved.metadata.get("improvements", [])
                }
            else:
                self.log("[ANSWER-EVOLUTION] ✗ Evolution produced no result")
                return {
                    "status": "error",
                    "message": "Evolution produced no result"
                }

        except Exception as e:
            self.log(f"[ANSWER-EVOLUTION] ✗ Error: {e}", "error")
            import traceback
            self.log(f"[ANSWER-EVOLUTION] {traceback.format_exc()}", "debug")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _evolve_answer_grounding(
        self,
        answer,
        requirement: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[RequirementAnswer]:
        """
        Evolve answer by grounding with more specific details and evidence.

        Grounding improvements:
        - Add more specific values, sequences, methods
        - Include additional supporting evidence
        - Reduce vagueness
        """
        self.log("  Applying grounding evolution...")

        # Extract review feedback if available
        review = self._get_answer_review(answer)
        weaknesses = []
        if review:
            weaknesses = review.get("weaknesses", [])
            weaknesses.extend(review.get("suggestions", []))

        # Build evolution prompt
        parsed_problem = self.config.get("parsed_problem", {})

        prompt = self.prompt_manager.get_prompt(
            "evolution/answer_evolution",
            parsed_problem=parsed_problem,
            requirement=requirement,
            parent_answer={
                "id": self._get_answer_id(answer),
                "answer": self._get_answer_content(answer),
                "rationale": self._get_answer_rationale(answer),
                "deliverables": self._get_answer_deliverables(answer),
                "review_weaknesses": weaknesses
            },
            context=self._format_context_for_prompt(context),
            evolution_method="grounding",
            focus="Add more specific details, concrete values, and supporting evidence"
        )

        return await self._generate_evolved_answer(answer, requirement, prompt, "grounding")

    async def _evolve_answer_coherence(
        self,
        answer,
        requirement: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[RequirementAnswer]:
        """
        Evolve answer by improving coherence with confirmed context.

        Coherence improvements:
        - Better alignment with dependency answers
        - Stronger logical connections
        - More consistent terminology
        """
        self.log("  Applying coherence evolution...")

        parsed_problem = self.config.get("parsed_problem", {})

        prompt = self.prompt_manager.get_prompt(
            "evolution/answer_evolution",
            parsed_problem=parsed_problem,
            requirement=requirement,
            parent_answer={
                "id": self._get_answer_id(answer),
                "answer": self._get_answer_content(answer),
                "rationale": self._get_answer_rationale(answer),
                "deliverables": self._get_answer_deliverables(answer)
            },
            context=self._format_context_for_prompt(context),
            evolution_method="coherence",
            focus="Improve alignment with confirmed context and strengthen logical connections"
        )

        return await self._generate_evolved_answer(answer, requirement, prompt, "coherence")

    async def _evolve_answer_simplification(
        self,
        answer,
        requirement: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[RequirementAnswer]:
        """
        Evolve answer by simplifying complexity while maintaining quality.

        Simplification improvements:
        - Remove unnecessary complexity
        - Focus on essential elements
        - Improve clarity and actionability
        """
        self.log("  Applying simplification evolution...")

        parsed_problem = self.config.get("parsed_problem", {})

        prompt = self.prompt_manager.get_prompt(
            "evolution/answer_evolution",
            parsed_problem=parsed_problem,
            requirement=requirement,
            parent_answer={
                "id": self._get_answer_id(answer),
                "answer": self._get_answer_content(answer),
                "rationale": self._get_answer_rationale(answer),
                "deliverables": self._get_answer_deliverables(answer)
            },
            context=self._format_context_for_prompt(context),
            evolution_method="simplification",
            focus="Remove unnecessary complexity while maintaining scientific rigor and completeness"
        )

        return await self._generate_evolved_answer(answer, requirement, prompt, "simplification")

    async def _evolve_answer_divergent(
        self,
        answer,
        requirement: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[RequirementAnswer]:
        """
        Evolve answer by exploring a different approach.

        Divergent improvements:
        - Consider alternative methodologies
        - Explore different perspectives
        - Challenge assumptions
        """
        self.log("  Applying divergent evolution...")

        parsed_problem = self.config.get("parsed_problem", {})

        prompt = self.prompt_manager.get_prompt(
            "evolution/answer_evolution",
            parsed_problem=parsed_problem,
            requirement=requirement,
            parent_answer={
                "id": self._get_answer_id(answer),
                "answer": self._get_answer_content(answer),
                "rationale": self._get_answer_rationale(answer),
                "deliverables": self._get_answer_deliverables(answer)
            },
            context=self._format_context_for_prompt(context),
            evolution_method="divergent",
            focus="Explore an alternative approach while remaining scientifically valid"
        )

        return await self._generate_evolved_answer(answer, requirement, prompt, "divergent")

    async def _generate_evolved_answer(
        self,
        parent_answer,
        requirement: Dict[str, Any],
        prompt: str,
        method: str
    ) -> Optional[RequirementAnswer]:
        """Common method to generate evolved answer from prompt"""
        if not self.llm:
            self.log("LLM not configured", "warning")
            return None

        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                purpose=f"evolve_answer_{method}"
            )

            req_id = self._get_answer_requirement_id(parent_answer)
            parent_id = self._get_answer_id(parent_answer)
            answer_id = f"ra_{req_id}_{uuid.uuid4().hex[:8]}"

            # Get requirement title
            if isinstance(requirement, dict):
                req_title = requirement.get("title", "")
            else:
                req_title = getattr(requirement, "title", "")

            evolved = RequirementAnswer(
                id=answer_id,
                requirement_id=req_id,
                requirement_title=req_title or (
                    parent_answer.requirement_title if hasattr(parent_answer, 'requirement_title')
                    else parent_answer.get("requirement_title", "")
                ),
                answer=response.get("answer", ""),
                rationale=response.get("rationale", ""),
                deliverables=response.get("deliverables", {}),
                confidence=max(0.0, min(1.0, float(response.get("confidence", 0.6)))),
                builds_on=self._get_answer_builds_on(parent_answer),
                status="generated",
                elo_rating=1200.0,
                parent_ids=[parent_id],
                evolution_method=method,
                iteration=self._get_answer_iteration(parent_answer) + 1,
                generated_at=datetime.now(),
                generation_method="evolution",
                metadata={
                    "parent_id": parent_id,
                    "evolution_method": method,
                    "improvements": response.get("improvements", []),
                    "comparison_to_parent": response.get("comparison_to_parent", "")
                }
            )

            return evolved

        except Exception as e:
            self.log(f"Error generating evolved answer: {e}", "error")
            return None

    def _format_context_for_prompt(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format confirmed context for prompt"""
        formatted = {}
        for dep_id, answer in context.items():
            if isinstance(answer, dict):
                formatted[dep_id] = {
                    "answer": answer.get("answer", "")[:400],
                    "deliverables": answer.get("deliverables", {})
                }
            else:
                formatted[dep_id] = {
                    "answer": getattr(answer, "answer", "")[:400],
                    "deliverables": getattr(answer, "deliverables", {})
                }
        return formatted
