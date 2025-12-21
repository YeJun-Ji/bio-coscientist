"""
Evolution Architect Agent - v5.0 Simplified Evaluation Pipeline

This agent enriches the SINGLE CONFIRMED ANSWER (tournament winner) without
changing core content.

Three Parallel Modules:
1. Protocol Generation: Step-by-step experimental validation procedure
2. Literature Support: Supporting/contradicting papers via PubMed
3. Risk Analysis: Potential issues + mitigation strategies

IMPORTANT: Only runs on the final confirmed answer, NOT on all generated answers.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core import RequirementAnswer
from ..core.evaluation_results import (
    EnrichmentResult,
    ProtocolModule,
    ExperimentalStep,
    LiteratureModule,
    Paper,
    RiskModule,
    Risk,
    MitigationStrategy
)
from ..external_apis import LLMClient
from ..memory import ContextMemory
from ..prompts import PromptManager
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EvolutionArchitectAgent(BaseAgent):
    """
    Enriches confirmed answer with protocol/literature/risk analysis.

    Architecture Note:
    - All 3 modules run in parallel (asyncio.gather)
    - Uses semaphore to limit concurrent MCP calls (max 2)
    - Runs in background (non-blocking)
    - Only applied to SINGLE confirmed answer
    """

    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        mcp_server_manager=None,
        **kwargs
    ):
        super().__init__(
            name="evolution_architect",
            memory=memory,
            config=config,
            llm_client=llm_client,
            mcp_server_manager=mcp_server_manager,
            **kwargs
        )
        self.prompt_manager = prompt_manager or PromptManager()
        self.mcp_manager = mcp_server_manager

        # Semaphore to limit concurrent MCP calls
        self.mcp_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent

        self.log("EvolutionArchitectAgent initialized (v5.0 - 3 parallel modules)")

    # ========================================================================
    # Main Entry Point (Required by BaseAgent)
    # ========================================================================

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to enrich_confirmed_answer.
        """
        answer = task.get("answer")
        requirement = task.get("requirement", {})
        context = task.get("context", {})

        enrichment_result = await self.enrich_confirmed_answer(
            answer=answer,
            requirement=requirement,
            context=context
        )

        return {
            "status": "success",
            "enrichment": enrichment_result.to_dict() if enrichment_result else None
        }

    # ========================================================================
    # Enrichment Logic
    # ========================================================================

    async def enrich_confirmed_answer(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any],
        context: Dict[str, RequirementAnswer]
    ) -> Optional[EnrichmentResult]:
        """
        Run 3 parallel enrichment modules on confirmed answer.

        Args:
            answer: CONFIRMED RequirementAnswer (tournament winner)
            requirement: Requirement specification
            context: Confirmed answers from dependencies

        Returns:
            EnrichmentResult with protocol, literature, risk modules

        Process:
            1. Launch 3 tasks in parallel: protocol, literature, risk
            2. Wait for all (with timeout protection)
            3. Aggregate results
            4. Store in answer.metadata["enrichment"]
        """
        self.log(f"Enriching confirmed answer: {answer.id}")

        # Launch all 3 modules concurrently
        protocol_task = asyncio.create_task(self.generate_protocol(answer))
        literature_task = asyncio.create_task(self.find_literature_support(answer))
        risk_task = asyncio.create_task(self.analyze_risks(answer))

        # Wait for all (with timeout protection)
        try:
            protocol, literature, risk = await asyncio.wait_for(
                asyncio.gather(protocol_task, literature_task, risk_task),
                timeout=120  # 2 minutes max
            )
        except asyncio.TimeoutError:
            self.log("Enrichment timeout - using partial results", level="warning")
            protocol = await protocol_task if protocol_task.done() else None
            literature = await literature_task if literature_task.done() else None
            risk = await risk_task if risk_task.done() else None
        except Exception as e:
            self.log(f"ERROR in enrichment: {e}", level="error")
            return None

        # Calculate composite confidence
        overall_confidence = self._calculate_composite_confidence(protocol, literature, risk)

        # Create enrichment result
        enrichment = EnrichmentResult(
            protocol=protocol,
            literature=literature,
            risk=risk,
            overall_confidence=overall_confidence
        )

        self.log(
            f"Enrichment complete: "
            f"protocol={'✓' if protocol else '✗'}, "
            f"literature={'✓' if literature else '✗'}, "
            f"risk={'✓' if risk else '✗'}, "
            f"confidence={overall_confidence:.2f}"
        )

        return enrichment

    # ========================================================================
    # Module 1: Protocol Generation
    # ========================================================================

    async def generate_protocol(self, answer: RequirementAnswer) -> Optional[ProtocolModule]:
        """
        Generate step-by-step experimental validation protocol.

        Args:
            answer: RequirementAnswer to generate protocol for

        Returns:
            ProtocolModule with steps, cost, duration, equipment
        """
        self.log(f"  Module 1: Generating protocol...")

        try:
            prompt = self.prompt_manager.get_prompt(
                "evolution/protocol_generation",
                answer={
                    "answer": answer.answer,
                    "deliverables": answer.deliverables
                }
            )

            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                purpose="evolution_protocol_generation"
            )

            steps = [
                ExperimentalStep(
                    step_number=step["step_number"],
                    action=step["action"],
                    duration=step["duration"],
                    materials=step["materials"],
                    expected_result=step["expected_result"]
                )
                for step in response.get("steps", [])
            ]

            return ProtocolModule(
                steps=steps,
                estimated_cost=response.get("estimated_cost"),
                estimated_duration=response.get("estimated_duration"),
                required_equipment=response.get("required_equipment", [])
            )

        except Exception as e:
            self.log(f"ERROR in protocol generation: {e}", level="error")
            return None

    # ========================================================================
    # Module 2: Literature Support
    # ========================================================================

    async def find_literature_support(self, answer: RequirementAnswer) -> Optional[LiteratureModule]:
        """
        Find supporting/contradicting papers via MCP PubMed tools.

        Args:
            answer: RequirementAnswer to find literature for

        Returns:
            LiteratureModule with supporting_papers, contradicting_papers, evidence_strength
        """
        self.log(f"  Module 2: Finding literature support...")

        try:
            # Step 1: Search PubMed via MCP (with semaphore protection)
            search_query = self._extract_search_query(answer)
            pubmed_results = await self._call_mcp_with_limit(
                "search_pubmed",
                {"query": search_query, "max_results": 10}
            )

            # Step 2: LLM categorizes papers as supporting/contradicting
            prompt = self.prompt_manager.get_prompt(
                "evolution/literature_support",
                answer={
                    "answer": answer.answer
                },
                papers=pubmed_results if pubmed_results else []
            )

            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                purpose="evolution_literature_support"
            )

            supporting_papers = [
                Paper(
                    title=p["title"],
                    pmid=p["pmid"],
                    relevance_score=p.get("relevance_score", 0.5),
                    key_finding=p.get("key_finding")
                )
                for p in response.get("supporting_papers", [])
            ]

            contradicting_papers = [
                Paper(
                    title=p["title"],
                    pmid=p["pmid"],
                    relevance_score=p.get("relevance_score", 0.5),
                    key_finding=p.get("key_finding")
                )
                for p in response.get("contradicting_papers", [])
            ]

            return LiteratureModule(
                supporting_papers=supporting_papers,
                contradicting_papers=contradicting_papers,
                evidence_strength=response.get("evidence_strength", "moderate")
            )

        except Exception as e:
            self.log(f"ERROR in literature support: {e}", level="error")
            return None

    async def _call_mcp_with_limit(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call MCP tool with semaphore protection.

        Args:
            tool_name: Name of MCP tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        async with self.mcp_semaphore:
            if not self.mcp_manager:
                self.log(f"MCP manager not available for {tool_name}", level="warning")
                return None

            try:
                # Call MCP tool via server manager
                result = await self.mcp_manager.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                self.log(f"ERROR calling MCP tool {tool_name}: {e}", level="error")
                return None

    def _extract_search_query(self, answer: RequirementAnswer) -> str:
        """
        Extract PubMed search query from answer.

        Args:
            answer: RequirementAnswer

        Returns:
            Search query string
        """
        # Simple extraction: Take first sentence of answer
        answer_text = answer.answer
        if ". " in answer_text:
            first_sentence = answer_text.split(". ")[0]
        else:
            first_sentence = answer_text[:200]  # First 200 chars

        # Clean up for search
        query = first_sentence.replace("\n", " ").strip()
        return query

    # ========================================================================
    # Module 3: Risk Analysis
    # ========================================================================

    async def analyze_risks(self, answer: RequirementAnswer) -> Optional[RiskModule]:
        """
        Analyze risks and mitigation strategies.

        Args:
            answer: RequirementAnswer to analyze risks for

        Returns:
            RiskModule with risks, mitigation_strategies, overall_risk_level
        """
        self.log(f"  Module 3: Analyzing risks...")

        try:
            prompt = self.prompt_manager.get_prompt(
                "evolution/risk_analysis",
                answer={
                    "answer": answer.answer,
                    "deliverables": answer.deliverables
                }
            )

            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                purpose="evolution_risk_analysis"
            )

            risks = [
                Risk(
                    type=r["type"],
                    description=r["description"],
                    likelihood=r["likelihood"],
                    impact=r["impact"]
                )
                for r in response.get("risks", [])
            ]

            mitigation_strategies = [
                MitigationStrategy(
                    for_risk=m["for_risk"],
                    strategy=m["strategy"],
                    effectiveness=m["effectiveness"]
                )
                for m in response.get("mitigation_strategies", [])
            ]

            return RiskModule(
                risks=risks,
                mitigation_strategies=mitigation_strategies,
                overall_risk_level=response.get("overall_risk_level", "medium")
            )

        except Exception as e:
            self.log(f"ERROR in risk analysis: {e}", level="error")
            return None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _calculate_composite_confidence(
        self,
        protocol: Optional[ProtocolModule],
        literature: Optional[LiteratureModule],
        risk: Optional[RiskModule]
    ) -> float:
        """
        Calculate composite confidence from all modules.

        Args:
            protocol: Protocol module result
            literature: Literature module result
            risk: Risk module result

        Returns:
            Confidence score (0.0-1.0)
        """
        scores = []

        # Protocol: Has steps?
        if protocol and protocol.steps:
            scores.append(0.8)
        elif protocol:
            scores.append(0.5)

        # Literature: Evidence strength
        if literature:
            strength_scores = {"strong": 0.9, "moderate": 0.6, "weak": 0.3}
            scores.append(strength_scores.get(literature.evidence_strength, 0.5))

        # Risk: Overall risk level (inverse)
        if risk:
            risk_scores = {"low": 0.9, "medium": 0.6, "high": 0.3}
            scores.append(risk_scores.get(risk.overall_risk_level, 0.5))

        # Average
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.5
