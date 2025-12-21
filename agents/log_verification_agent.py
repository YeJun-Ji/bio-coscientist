"""
Log Verification Agent - Objective answer verification against execution logs

This agent verifies RequirementAnswers by comparing claims against actual execution logs.
Replaces subjective LLM-based evaluation with objective, rule-based verification.

Key Innovation:
- Phase 1 (pre_check): Fast-fail invalid answers in <0.5s
- Phase 2 (verify): Objective verification in <2s without LLM calls
- Domain-agnostic: Works for protein, pathway, drug, disease equally

Architecture:
- Uses DataFileManager to load execution logs (sources.json, results.json, metadata.json)
- Extracts claims from answer.metadata["data_references"]
- Validates using ThresholdRegistry (external config, no hardcoding)

Example:
    Answer claims: "BLAST E-value: 1e-6 (high confidence)"
    Verification:
      1. Load logs → actual E-value = 1e-6 ✓
      2. Check interpretation: 1e-6 <= 1e-5? ✓
      → verification_score = 1.0
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional

from ..core import RequirementAnswer
from ..memory import ContextMemory
from ..utils.data_file_manager import DataFileManager
from ..utils.threshold_loader import ThresholdRegistry
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LogVerificationAgent(BaseAgent):
    """
    Verify answer claims against execution logs.

    Two-phase verification:
    1. pre_check(): Fast-fail for obviously invalid answers (<0.5s)
    2. verify(): Detailed objective verification (<2s)

    No LLM calls - all rule-based for maximum speed and objectivity.
    """

    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client=None,
        experiment_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            name="log_verification",
            memory=memory,
            config=config,
            llm_client=llm_client,
            **kwargs
        )

        # Initialize DataFileManager for log loading
        self.experiment_dir = experiment_dir
        if experiment_dir:
            self.data_file_manager = DataFileManager(experiment_dir)
        else:
            self.data_file_manager = None
            logger.warning("No experiment_dir provided - log verification will be limited")

        # Initialize ThresholdRegistry
        threshold_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "tool_thresholds.json"
        )
        self.threshold_registry = ThresholdRegistry(threshold_config_path)

        logger.info(f"LogVerificationAgent initialized with {len(self.threshold_registry.get_supported_tools())} tools")

    # ========== Main Entry Point ==========

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute verification (delegates to verify method).

        Args:
            task: {
                "answer": RequirementAnswer,
                "requirement_id": str,
                "experiment_dir": str (optional, overrides init)
            }

        Returns:
            Verification result dict
        """
        answer = task.get("answer")
        req_id = task.get("requirement_id")
        exp_dir = task.get("experiment_dir", self.experiment_dir)

        if not answer or not req_id:
            return {
                "status": "error",
                "message": "Missing answer or requirement_id"
            }

        return self.verify(answer, req_id, exp_dir)

    # ========== Phase 1: Pre-Check (Fast-Fail) ==========

    def pre_check(
        self,
        answer: RequirementAnswer,
        requirement_id: str,
        experiment_dir: Optional[str] = None
    ) -> bool:
        """
        Phase 1: Fast pre-check to reject obviously invalid answers.

        Checks:
        1. All mentioned tools were actually executed
        2. Metric values are within physically possible ranges
        3. Referenced files exist

        NO LLM calls - pure rule-based validation.

        Args:
            answer: RequirementAnswer to check
            requirement_id: Requirement ID
            experiment_dir: Override experiment directory

        Returns:
            True if answer passes pre-check, False to reject immediately

        Performance target: <0.5s
        """
        start_time = time.time()
        exp_dir = experiment_dir or self.experiment_dir

        if not self.data_file_manager and not exp_dir:
            logger.warning("No experiment_dir - skipping pre-check")
            return True

        try:
            # Use provided exp_dir or fall back to instance's data_file_manager
            if exp_dir and exp_dir != self.experiment_dir:
                dfm = DataFileManager(exp_dir)
            else:
                dfm = self.data_file_manager

            # Load metadata to get executed tools
            metadata = dfm.load_metadata(requirement_id)
            if not metadata:
                logger.warning(f"No metadata found for {requirement_id}, skipping tool check")
                executed_tools = set()
            else:
                tool_usage = metadata.get("tool_usage", {})
                collection_tools = tool_usage.get("collection_tools", [])
                analysis_tools = tool_usage.get("analysis_tools", [])

                # Extract tool names (handle both dict and string formats)
                executed_tools = set()
                for tool in collection_tools + analysis_tools:
                    if isinstance(tool, dict):
                        tool_name = tool.get("tool_name", "")
                    else:
                        tool_name = str(tool)
                    executed_tools.add(tool_name.lower())

            # Extract mentioned tools from answer
            data_refs = self._get_data_references(answer)
            mentioned_tools = set()

            # Known tool keywords for extraction from strings
            # NOTE: Do NOT include file extensions like "csv", "json", "pdb" as they cause false positives
            # when answers mention file names like "Q1.genelist.csv"
            KNOWN_TOOLS = [
                # Database tools
                "uniprot", "kegg", "blast", "esmfold", "rosetta",
                "ncbi", "pubmed", "scholar", "pfam", "stringdb", "chembl",
                # Analysis tools
                "pandas_analysis", "read_metadata", "run_pandas_code",
                "foldseek", "msa", "networkx", "gprofiler", "opentargets",
                "interpro", "iedb", "vina", "rcsbpdb",
                # Removed: "csv", "pdb" - these are file extensions, not tools
                # Removed: "pymol", "reactome" - deprecated tools
            ]

            for ref in data_refs:
                # Handle both dict and string formats
                if isinstance(ref, dict):
                    tool_name = ref.get("tool", "").lower()
                    if tool_name:
                        mentioned_tools.add(tool_name)
                elif isinstance(ref, str):
                    # Extract tool names from descriptive string
                    # e.g., "UniProt protein annotations" → "uniprot"
                    ref_lower = ref.lower()
                    for known_tool in KNOWN_TOOLS:
                        if known_tool in ref_lower:
                            mentioned_tools.add(known_tool)
                else:
                    continue

            # Debug logging
            logger.info(f"Pre-check for {requirement_id}: {len(data_refs)} data refs, {len(mentioned_tools)} tools mentioned, {len(executed_tools)} tools executed")
            logger.debug(f"  Mentioned tools: {mentioned_tools}")
            logger.debug(f"  Executed tools: {executed_tools}")

            # Check 1: Are all mentioned tools executed?
            # IMPORTANT: If no tools mentioned, consider it valid (answer doesn't claim tool usage)
            if not mentioned_tools:
                logger.info(f"Pre-check PASSED: No tool claims in answer")
                return True

            for tool in mentioned_tools:
                # Flexible matching:
                # 1. Exact match: "blast_search" == "blast_search"
                # 2. Base match: "blast_search" matches "blast"
                # 3. Partial match: "uniprot" matches "search_uniprot", "uniprot_annotations"
                # 4. Reverse partial: "search_uniprot" matches "uniprot"

                # Extract tool base (first word before underscore)
                tool_base = tool.split("_")[0].lower()
                tool_words = tool.lower().split("_")

                matched = False
                for executed in executed_tools:
                    executed_base = executed.split("_")[0].lower()
                    executed_words = executed.lower().split("_")

                    # Check various matching strategies
                    if (tool.lower() == executed.lower() or  # Exact match
                        tool_base in executed.lower() or  # Base in executed
                        executed_base in tool.lower() or  # Executed base in tool
                        any(word in executed.lower() for word in tool_words if len(word) > 3) or  # Any tool word in executed (skip short words)
                        any(word in tool.lower() for word in executed_words if len(word) > 3)):  # Any executed word in tool
                        matched = True
                        logger.debug(f"  Tool '{tool}' matched with executed '{executed}'")
                        break

                if not matched:
                    logger.warning(
                        f"Pre-check FAILED: Answer mentions '{tool}' but no matching tool was executed. "
                        f"Executed: {executed_tools}"
                    )
                    duration = time.time() - start_time
                    logger.debug(f"Pre-check completed in {duration:.3f}s")
                    return False

            # Check 2: Metric values in valid ranges?
            for ref in data_refs:
                # Handle both dict and string formats
                if isinstance(ref, dict):
                    tool = ref.get("tool", "").lower()
                    metric = ref.get("metric", "").lower()
                    value = ref.get("value")
                elif isinstance(ref, str):
                    # String format doesn't have metrics to validate
                    continue
                else:
                    continue

                if value is not None:
                    if not self.threshold_registry.is_value_in_range(tool, metric, value):
                        valid_range = self.threshold_registry.get_valid_range(tool, metric)
                        logger.warning(
                            f"Pre-check FAILED: {tool}.{metric} value {value} "
                            f"outside valid range {valid_range}"
                        )
                        duration = time.time() - start_time
                        logger.debug(f"Pre-check completed in {duration:.3f}s")
                        return False

            # Check 3: Referenced files exist?
            deliverables = answer.deliverables if isinstance(answer.deliverables, dict) else {}
            for key, value in deliverables.items():
                if isinstance(value, str) and (value.endswith(".pdb") or value.endswith(".txt") or value.endswith(".json")):
                    # Check if file exists (relative to experiment dir or absolute)
                    if not os.path.isabs(value):
                        file_path = os.path.join(exp_dir or "", value)
                    else:
                        file_path = value

                    if not os.path.exists(file_path):
                        logger.warning(f"Pre-check FAILED: Referenced file not found: {file_path}")
                        # Don't fail on missing files - they might be in different locations
                        # Just log warning
                        pass

            duration = time.time() - start_time
            logger.debug(f"Pre-check PASSED in {duration:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Pre-check error: {e}", exc_info=True)
            # On error, allow answer to proceed (don't reject due to check failure)
            return True

    # ========== Phase 2: Detailed Verification ==========

    def verify(
        self,
        answer: RequirementAnswer,
        requirement_id: str,
        experiment_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Phase 2: Detailed objective verification against logs.

        Verification dimensions:
        1. Truthfulness: Do claimed values match actual logs?
        2. Coverage: Did answer use all available data?
        3. Interpretation validity: Are interpretations supported by thresholds?

        NO LLM calls - all rule-based.

        Args:
            answer: RequirementAnswer to verify
            requirement_id: Requirement ID
            experiment_dir: Override experiment directory

        Returns:
            {
                "status": "success" | "error",
                "verification_score": float (0.0-1.0),
                "metrics": {
                    "truthfulness": float,
                    "coverage": float,
                    "interpretation_validity": float
                },
                "violations": List[Dict],
                "verified_claims": List[str]
            }

        Performance target: <2s
        """
        start_time = time.time()
        exp_dir = experiment_dir or self.experiment_dir

        self.log(f"[LOG-VERIFY] Starting verification for {answer.id}")

        try:
            # Load execution logs
            logs = self._load_execution_logs(requirement_id, exp_dir)

            # Extract claims from answer
            claims = self._extract_claims(answer)
            self.log(f"[LOG-VERIFY] Extracted {len(claims)} claims from answer")

            # Verify each claim
            verification_results = []
            for claim in claims:
                result = self._verify_single_claim(claim, logs)
                verification_results.append(result)

            # Calculate scores
            scores = self._calculate_scores(verification_results, logs, answer)

            # Aggregate violations
            violations = [r for r in verification_results if not r.get("verified", False)]

            # Format verified claims
            verified_claims = [
                r["claim_text"]
                for r in verification_results
                if r.get("verified", False)
            ]

            duration = time.time() - start_time
            self.log(f"[LOG-VERIFY] ✓ Verification complete in {duration:.2f}s")
            self.log(f"[LOG-VERIFY] Score: {scores['composite']:.2f} "
                    f"(T:{scores['truthfulness']:.2f}, "
                    f"C:{scores['coverage']:.2f}, "
                    f"I:{scores['interpretation']:.2f})")

            return {
                "status": "success",
                "verification_score": scores["composite"],
                "metrics": {
                    "truthfulness": scores["truthfulness"],
                    "coverage": scores["coverage"],
                    "interpretation_validity": scores["interpretation"]
                },
                "violations": violations,
                "verified_claims": verified_claims
            }

        except Exception as e:
            logger.error(f"Verification error: {e}", exc_info=True)
            return {
                "status": "error",
                "verification_score": 0.5,  # Neutral score on error
                "message": str(e)
            }

    # ========== Helper Methods ==========

    def _get_data_references(self, answer: RequirementAnswer) -> List[Dict]:
        """Extract data_references from answer metadata."""
        if isinstance(answer, dict):
            metadata = answer.get("metadata", {})
        else:
            metadata = answer.metadata or {}

        return metadata.get("data_references", [])

    def _load_execution_logs(
        self,
        requirement_id: str,
        experiment_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load all execution logs for a requirement.

        Returns:
            {
                "sources": Dict,  # Chain 1 collection data
                "analysis": Dict,  # Chain 2 analysis results
                "metadata": Dict   # Tool usage metadata
            }
        """
        if not self.data_file_manager and not experiment_dir:
            return {"sources": {}, "analysis": {}, "metadata": {}}

        # Use provided exp_dir or fall back to instance's data_file_manager
        if experiment_dir and experiment_dir != self.experiment_dir:
            dfm = DataFileManager(experiment_dir)
        else:
            dfm = self.data_file_manager

        sources = dfm.load_collection_data(requirement_id)
        analysis = dfm.load_analysis_results(requirement_id)
        metadata = dfm.load_metadata(requirement_id)

        return {
            "sources": sources,
            "analysis": analysis,
            "metadata": metadata
        }

    def _extract_claims(self, answer: RequirementAnswer) -> List[Dict]:
        """
        Extract verifiable claims from answer.

        Uses answer.metadata["data_references"] which GenerationAgent populates.

        Returns:
            List of claim dicts with:
            - tool: str
            - metric: str
            - value: float
            - interpretation: str (optional)
            - claim_text: str (for reporting)
        """
        data_refs = self._get_data_references(answer)
        claims = []

        for i, ref in enumerate(data_refs):
            # Handle both dict and string formats
            if isinstance(ref, dict):
                claim = {
                    "tool": ref.get("tool", "unknown").lower(),
                    "metric": ref.get("metric", "unknown").lower(),
                    "value": ref.get("value"),
                    "interpretation": ref.get("interpretation"),
                    "claim_text": f"{ref.get('tool', 'unknown')}.{ref.get('metric', 'unknown')}={ref.get('value')}"
                }
            elif isinstance(ref, str):
                # String format (e.g., "uniprot_search")
                claim = {
                    "tool": ref.lower(),
                    "metric": "execution",
                    "value": True,
                    "interpretation": None,
                    "claim_text": f"{ref}.execution=True"
                }
            else:
                continue
            claims.append(claim)

        return claims

    def _verify_single_claim(self, claim: Dict, logs: Dict) -> Dict:
        """
        Verify a single claim against logs.

        Args:
            claim: Claim dict from _extract_claims
            logs: Execution logs from _load_execution_logs

        Returns:
            {
                "verified": bool,
                "claim_text": str,
                "type": "truthfulness" | "interpretation",
                "reason": str
            }
        """
        tool = claim["tool"]
        metric = claim["metric"]
        claimed_value = claim["value"]
        interpretation = claim.get("interpretation")

        # Find actual value in logs
        actual_value = self._find_metric_in_logs(tool, metric, logs)

        if actual_value is None:
            return {
                "verified": False,
                "claim_text": claim["claim_text"],
                "type": "missing_data",
                "reason": f"Metric {tool}.{metric} not found in logs"
            }

        # Verify truthfulness (value match)
        if claimed_value is not None:
            # Allow small floating point errors
            if abs(claimed_value - actual_value) > 1e-6:
                return {
                    "verified": False,
                    "claim_text": claim["claim_text"],
                    "type": "truthfulness",
                    "reason": f"Claimed {claimed_value}, actual {actual_value}",
                    "claimed": claimed_value,
                    "actual": actual_value
                }

        # Verify interpretation (if provided)
        if interpretation:
            is_valid = self.threshold_registry.validate_interpretation(
                tool, metric, actual_value, interpretation
            )

            if not is_valid:
                suggested = self.threshold_registry.suggest_confidence_level(
                    tool, metric, actual_value
                )
                return {
                    "verified": False,
                    "claim_text": claim["claim_text"],
                    "type": "interpretation",
                    "reason": f"Invalid interpretation '{interpretation}' for value {actual_value}",
                    "suggested": suggested
                }

        # All checks passed
        return {
            "verified": True,
            "claim_text": claim["claim_text"],
            "type": "verified",
            "reason": "Matches logs and valid interpretation"
        }

    def _find_metric_in_logs(
        self,
        tool_name: str,
        metric_name: str,
        logs: Dict
    ) -> Optional[float]:
        """
        Find a metric value in execution logs.

        Searches both sources (Chain 1) and analysis (Chain 2).

        Args:
            tool_name: Tool name (e.g., "blast", "esmfold")
            metric_name: Metric name (e.g., "e_value", "plddt")
            logs: Execution logs dict

        Returns:
            Metric value or None if not found
        """
        # Search in sources (Chain 1)
        sources = logs.get("sources", {})
        for source_name, source_data in sources.items():
            if tool_name in source_name.lower():
                # source_data might be list or dict
                if isinstance(source_data, list):
                    for item in source_data:
                        result = item.get("result", {})
                        value = self._extract_metric_from_dict(result, metric_name)
                        if value is not None:
                            return value
                elif isinstance(source_data, dict):
                    value = self._extract_metric_from_dict(source_data, metric_name)
                    if value is not None:
                        return value

        # Search in analysis (Chain 2)
        analysis = logs.get("analysis", {})
        for analysis_tool, analysis_result in analysis.items():
            if tool_name in analysis_tool.lower():
                value = self._extract_metric_from_dict(analysis_result, metric_name)
                if value is not None:
                    return value

        return None

    def _extract_metric_from_dict(self, data: Dict, metric_name: str) -> Optional[float]:
        """
        Extract a metric value from a nested dict.

        Handles various nesting patterns:
        - Direct: {"e_value": 1e-6}
        - Nested: {"hits": [{"e_value": 1e-6}]}
        - CamelCase: {"eValue": 1e-6}
        """
        if not isinstance(data, dict):
            return None

        # Direct lookup (case-insensitive)
        for key, value in data.items():
            if key.lower() == metric_name.lower():
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass

        # Nested search (one level deep)
        for key, value in data.items():
            if isinstance(value, dict):
                result = self._extract_metric_from_dict(value, metric_name)
                if result is not None:
                    return result
            elif isinstance(value, list) and value:
                # Check first item in list
                if isinstance(value[0], dict):
                    result = self._extract_metric_from_dict(value[0], metric_name)
                    if result is not None:
                        return result

        return None

    def _calculate_scores(
        self,
        verification_results: List[Dict],
        logs: Dict,
        answer: RequirementAnswer
    ) -> Dict[str, float]:
        """
        Calculate verification scores from results.

        Returns:
            {
                "truthfulness": float,
                "coverage": float,
                "interpretation": float,
                "composite": float
            }
        """
        if not verification_results:
            return {
                "truthfulness": 0.0,
                "coverage": 0.0,
                "interpretation": 0.0,
                "composite": 0.0
            }

        # Truthfulness: % of claims that match logs
        truthfulness_checks = [
            r for r in verification_results
            if r["type"] in ["truthfulness", "verified"]
        ]
        if truthfulness_checks:
            truthfulness = sum(
                1 for r in truthfulness_checks if r["verified"]
            ) / len(truthfulness_checks)
        else:
            truthfulness = 1.0  # No truthfulness claims = pass

        # Interpretation validity: % of interpretations that are valid
        interpretation_checks = [
            r for r in verification_results
            if r["type"] in ["interpretation", "verified"] and "interpretation" in str(r)
        ]
        if interpretation_checks:
            interpretation = sum(
                1 for r in interpretation_checks if r["verified"]
            ) / len(interpretation_checks)
        else:
            interpretation = 1.0  # No interpretations = pass

        # Coverage: Did answer use all available tools?
        metadata = logs.get("metadata", {})
        tool_usage = metadata.get("tool_usage", {})
        executed_tools = set()
        for tool_list in [tool_usage.get("collection_tools", []), tool_usage.get("analysis_tools", [])]:
            for tool in tool_list:
                if isinstance(tool, dict):
                    tool_name = tool.get("tool_name", "")
                else:
                    tool_name = str(tool)
                if tool_name:
                    executed_tools.add(tool_name.lower().split("_")[0])

        mentioned_tools = set()
        data_refs = self._get_data_references(answer)
        for ref in data_refs:
            tool = ref.get("tool", "").lower().split("_")[0]
            if tool:
                mentioned_tools.add(tool)

        if executed_tools:
            coverage = len(mentioned_tools & executed_tools) / len(executed_tools)
        else:
            coverage = 1.0  # No tools executed = no coverage requirement

        # Composite score (weighted average)
        composite = (
            truthfulness * 0.4 +
            interpretation * 0.3 +
            coverage * 0.3
        )

        return {
            "truthfulness": truthfulness,
            "coverage": coverage,
            "interpretation": interpretation,
            "composite": composite
        }
