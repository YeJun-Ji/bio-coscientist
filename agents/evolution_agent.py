"""
Evolution Agent - Post-Confirmation Enrichment System (v3.0)

⚠️ DEPRECATED in v5.0: Use EvolutionArchitectAgent instead.
⚠️ This agent will be removed in v6.0.

NEW ROLE: After a RequirementAnswer is confirmed, this agent runs validation
tools asynchronously to add credibility metrics and confidence scores.

LEGACY ROLE (deprecated): Pre-confirmation answer improvement via grounding,
coherence, etc. Legacy methods kept as _legacy_* for reference.
"""

import json
import logging
import uuid
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import ResearchGoal, RequirementAnswer
from ..external_apis import LLMClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """
    Post-Confirmation Enrichment Agent (v3.0)

    NEW ROLE:
    After answer confirmation, executes ALL available validation tools asynchronously
    to add credibility metrics (BLAST E-values, ESMFold pLDDT, etc.) and confidence
    scores to confirmed answers.

    DEPRECATED ROLE:
    Pre-confirmation answer improvement via grounding, coherence, simplification,
    divergent methods. These are kept as _legacy_* methods for reference.
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

    # ========== Main Entry Point ==========

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to enrich_confirmed_answer for post-confirmation enrichment.
        """
        # DEPRECATED v5.0: Warn users to migrate to EvolutionArchitectAgent
        warnings.warn(
            "EvolutionAgent is deprecated in v5.0. "
            "Use EvolutionArchitectAgent for structured enrichment (protocol/literature/risk). "
            "This agent will be removed in v6.0.",
            DeprecationWarning,
            stacklevel=2
        )

        return await self.enrich_confirmed_answer(task)

    # ========== Post-Confirmation Enrichment (v3.0) ==========

    async def enrich_confirmed_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-confirmation enrichment: Run validation tools to add credibility metrics.

        Workflow:
        1. Select enrichment tools (LLM-based, using answer content)
        2. Execute tools in parallel (asyncio.gather)
        3. Compile enrichment data (extract metrics)
        4. Update answer.metadata["enrichments"]

        Args:
            task: {"answer": RequirementAnswer (confirmed), "requirement": Dict}

        Returns:
            {"status": "success", "enrichments": Dict, "tools_executed": List[str]}
        """
        import time
        func_start = time.time()

        answer = task.get("answer")
        requirement = task.get("requirement", {})

        if not answer:
            self.log("ERROR: 'answer' not provided in task", "error")
            return {"status": "error", "message": "answer not provided"}

        answer_id = self._get_answer_id(answer)
        self.log(f"[POST-CONFIRM-ENRICH] Enriching confirmed answer {answer_id}")

        try:
            # Step 1: Select enrichment tools
            tool_plan = await self._select_enrichment_tools(answer, requirement)
            self.log(f"  Selected {len(tool_plan)} enrichment tools")

            if not tool_plan:
                self.log("  No enrichment tools selected - skipping enrichment")
                return {
                    "status": "success",
                    "enrichments": {},
                    "tools_executed": []
                }

            # Step 2: Execute tools in parallel
            from ..tools.executor import ToolExecutor
            if not hasattr(self, 'tool_executor'):
                self.tool_executor = ToolExecutor(
                    self.tool_registry,
                    self.mcp_manager,  # BaseAgent uses 'mcp_manager', not 'mcp_server_manager'
                    self.llm
                )

            enrichment_results = await self.tool_executor.execute_tool_calls(
                tool_calls=tool_plan,
                parallel=True
            )

            # Step 3: Compile enrichment data
            enrichment_data = self._compile_enrichment_data(enrichment_results)

            # Step 4: Update answer metadata
            if isinstance(answer, dict):
                if "metadata" not in answer:
                    answer["metadata"] = {}
                answer["metadata"]["enrichments"] = enrichment_data
            else:
                if not answer.metadata:
                    answer.metadata = {}
                answer.metadata["enrichments"] = enrichment_data

            tools_executed = [r["name"] for r in enrichment_results if r.get("status") == "success"]
            func_duration = time.time() - func_start
            self.log(f"[POST-CONFIRM-ENRICH] ✓ Complete: {len(tools_executed)} tools executed ({func_duration:.2f}s)")

            return {
                "status": "success",
                "enrichments": enrichment_data,
                "tools_executed": tools_executed
            }

        except Exception as e:
            self.log(f"[POST-CONFIRM-ENRICH] ✗ Error: {e}", "error")
            import traceback
            self.log(f"[POST-CONFIRM-ENRICH] {traceback.format_exc()}", "debug")
            return {
                "status": "error",
                "answer_id": answer_id,
                "message": str(e)
            }

    async def _select_enrichment_tools(self, answer, requirement) -> List[Dict[str, Any]]:
        """
        Use LLM to select validation tools based on answer content.

        Returns: [{"name": "blast_search", "arguments": {...}}, ...]
        """
        if not self.tool_registry:
            self.log("  Tool registry not available - cannot select tools", "warning")
            return []

        # Get all available tools from registry
        available_tools = self.tool_registry.get_mcp_tools()
        if not available_tools:
            self.log("  No tools available in registry", "warning")
            return []

        available_tools_list = [
            f"- {tool.name}: {tool.description}"
            for tool in available_tools
        ]

        # Build prompt
        prompt = self.prompt_manager.get_prompt(
            "evolution/select_enrichment_tools",
            answer={
                "content": self._get_answer_content(answer),
                "deliverables": self._get_answer_deliverables(answer)
            },
            requirement=requirement,
            available_tools="\n".join(available_tools_list)
        )

        try:
            # LLM selects tools
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                purpose="select_enrichment_tools"
            )

            tools = response.get("tools", [])
            self.log(f"  LLM selected {len(tools)} tools")
            return tools

        except Exception as e:
            self.log(f"  Error selecting enrichment tools: {e}", "error")
            return []

    def _compile_enrichment_data(self, enrichment_results: List[Dict]) -> Dict[str, Any]:
        """
        Extract metrics from tool results and compile into structured enrichment data.

        Extensible: Add elif blocks for new tools without code changes.

        Returns:
            {
                "validation_metrics": {"blast_e_value": 1e-10, "esmfold_plddt": 85.3},
                "confidence_scores": {"overall": 0.88, "sequence_validation": 0.95},
                "additional_evidence": ["BLAST found 15 homologs...", ...],
                "tool_results": {"blast_search": {...}, "fold_sequence": {...}}
            }
        """
        enrichment_data = {
            "validation_metrics": {},
            "confidence_scores": {},
            "additional_evidence": [],
            "tool_results": {}
        }

        for result in enrichment_results:
            if result.get("status") != "success":
                continue

            tool_name = result["name"]
            tool_result = result.get("result", {})
            enrichment_data["tool_results"][tool_name] = tool_result

            # Tool-specific metric extraction (EXTENSIBLE - add new tools here!)

            # BLAST or similar protein search
            if tool_name in ["blast_search", "find_similar_proteins"]:
                hits = tool_result.get("hits", [])
                if hits:
                    best_hit = hits[0]
                    e_value = float(best_hit.get("evalue", best_hit.get("e_value", 1.0)))
                    enrichment_data["validation_metrics"]["blast_e_value"] = e_value
                    enrichment_data["additional_evidence"].append(
                        f"BLAST validation: E-value={e_value:.2e}, {len(hits)} homologs found"
                    )
                    # Confidence: E < 1e-5 = high
                    enrichment_data["confidence_scores"]["sequence_validation"] = (
                        1.0 if e_value < 1e-5 else (0.5 if e_value < 1e-3 else 0.2)
                    )

            # ESMFold structure prediction
            elif tool_name in ["fold_sequence", "esmfold_predict"]:
                plddt_scores = tool_result.get("plddt", [])
                if plddt_scores:
                    mean_plddt = sum(plddt_scores) / len(plddt_scores)
                    enrichment_data["validation_metrics"]["esmfold_mean_plddt"] = mean_plddt
                    enrichment_data["additional_evidence"].append(
                        f"ESMFold structure prediction: mean pLDDT={mean_plddt:.1f}"
                    )
                    # Confidence: >= 70 = high
                    enrichment_data["confidence_scores"]["structure_validation"] = (
                        min(1.0, mean_plddt / 90.0) if mean_plddt >= 50 else 0.3
                    )

            # Rosetta energy calculation
            elif tool_name in ["calculate_rosetta_energy", "rosetta_score"]:
                total_score = tool_result.get("total_score")
                if total_score is not None:
                    enrichment_data["validation_metrics"]["rosetta_total_score"] = total_score
                    enrichment_data["additional_evidence"].append(
                        f"Rosetta energy: total_score={total_score:.2f}"
                    )
                    # Confidence: negative score = favorable
                    enrichment_data["confidence_scores"]["stability_validation"] = (
                        min(1.0, max(0.0, (-total_score + 50) / 100))
                    )

            # ADD NEW TOOLS HERE without code changes!
            # Example:
            # elif tool_name == "new_validation_tool":
            #     enrichment_data["validation_metrics"]["new_metric"] = ...

        # Calculate overall confidence
        if enrichment_data["confidence_scores"]:
            enrichment_data["confidence_scores"]["overall"] = sum(
                enrichment_data["confidence_scores"].values()
            ) / len(enrichment_data["confidence_scores"])

        return enrichment_data
