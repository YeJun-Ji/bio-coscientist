"""
Generation Agent - Generates RequirementAnswers for Sequential Confirmation

This agent is responsible for:
1. Generating diverse answers for each Requirement (run_for_requirement)
2. Evolving existing answers for improvement (generate_evolved_answer)
3. Collecting data from MCP servers (KEGG, UniProt, PDB) for evidence-based generation

Architecture Note - MCP Server vs Tool:
- MCP Server: External process (e.g., KEGG-MCP-Server) that provides multiple tools
- Tool: Individual function that LLM can select via function calling (e.g., search_kegg_pathway)
- ToolRegistry: Catalog of all available Tools across all MCP Servers
- ToolExecutor: Routes Tool calls to appropriate MCP Servers
- LLM receives Tool definitions, selects which Tools to call, ToolExecutor handles execution
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import ResearchGoal, RequirementAnswer, Requirement
from ..external_apis import LLMClient
from ..memory import ContextMemory
from ..tools import ToolRegistry, ToolExecutor
from ..prompts import PromptManager
from .base_agent import BaseAgent
import uuid

logger = logging.getLogger(__name__)




class GenerationAgent(BaseAgent):
    """
    Generates novel hypotheses and research proposals.
    
    Techniques:
    - Data-driven hypothesis generation using KEGG, UniProt, PDB (via MCP Servers)
    - LLM function calling with dynamic Tool selection
    - Iterative assumptions identification
    - Research expansion based on existing hypotheses
    """
    
    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        prompt_manager: Optional[PromptManager] = None,
        mcp_server_manager=None,
        experiment_dir: Optional[str] = None,  # NEW parameter
        **kwargs
    ):
        super().__init__(
            name="generation",
            memory=memory,
            config=config,
            llm_client=llm_client,
            mcp_server_manager=mcp_server_manager,
            **kwargs
        )

        # Tool management (MCP servers provide data access)
        self.tool_registry = tool_registry or ToolRegistry()
        self.mcp_manager = mcp_server_manager
        self.tool_executor = ToolExecutor(
            self.tool_registry,
            mcp_server_manager=self.mcp_manager
        )

        # Prompt management
        self.prompt_manager = prompt_manager or PromptManager()

        # === NEW: Data file management ===
        self.data_file_manager = None
        if experiment_dir:
            from ..utils.data_file_manager import DataFileManager
            self.data_file_manager = DataFileManager(experiment_dir)
            self.log(f"DataFileManager initialized for: {experiment_dir}")

        self.log(f"GenerationAgent initialized with ToolRegistry ({len(self.tool_registry._tools)} tools) and PromptManager")

    # ========================================================================
    # Main Entry Point (Required by BaseAgent)
    # ========================================================================

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to run_for_requirement for RequirementAnswer generation.
        """
        return await self.run_for_requirement(task)

    # ========================================================================
    # NEW: RequirementAnswer-based Generation (Sequential Confirmation)
    # ========================================================================

    async def run_for_requirement(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate multiple diverse answers for a single Requirement.

        This is the new entry point for Sequential Confirmation architecture.
        Instead of generating entire hypotheses, this generates N diverse answers
        for a specific requirement, using confirmed answers from dependencies as context.

        Args:
            task: {
                "requirement": Dict or Requirement,  # The requirement to answer
                "num_answers": int,                  # Number of diverse answers to generate (default: 3)
                "context": Dict[str, RequirementAnswer],  # Confirmed answers from dependencies
                "research_goal": ResearchGoal (optional, falls back to memory)
            }

        Returns:
            {
                "status": "success" | "error",
                "requirement_id": str,
                "answers": List[RequirementAnswer],
                "data_sources": Dict
            }
        """
        import time
        func_start = time.time()

        self.log("=" * 60)
        self.log("🚀 GENERATION AGENT: run_for_requirement()")
        self.log("=" * 60)

        # Extract task parameters
        requirement = task.get("requirement")
        if not requirement:
            self.log("ERROR: 'requirement' not provided in task", "error")
            return {"status": "error", "message": "requirement not provided"}

        # Support both dict and Requirement object
        if isinstance(requirement, dict):
            req_id = requirement.get("requirement_id", requirement.get("step_id", "unknown"))
            req_title = requirement.get("title", "")
        else:
            req_id = getattr(requirement, "requirement_id", "unknown")
            req_title = getattr(requirement, "title", "")

        num_answers = task.get("num_answers", 3)
        context = task.get("context", {})  # Dict[str, RequirementAnswer]

        self.log(f"[REQ-GEN] Requirement: {req_id} - {req_title[:50]}...")
        self.log(f"[REQ-GEN] Generating {num_answers} diverse answers")
        self.log(f"[REQ-GEN] Context: {len(context)} confirmed dependencies")

        # Get research goal from task or memory
        research_goal = task.get("research_goal") or self.memory.get_research_goal()
        if not research_goal:
            self.log("ERROR: research_goal not available", "error")
            return {"status": "error", "message": "research_goal not available"}

        try:
            # Step 1: Collect data relevant to this requirement (2-Chain)
            self.log("[REQ-GEN] Step 1/3: Collecting data for requirement...")
            step1_start = time.time()
            collected_data = await self._collect_data_for_requirement(
                requirement=requirement,
                research_goal=research_goal,
                context=context
            )
            step1_duration = time.time() - step1_start
            self.log(f"[REQ-GEN] ✓ Step 1 complete ({step1_duration:.2f}s)")

            # Step 2: Generate N diverse answers
            self.log(f"[REQ-GEN] Step 2/3: Generating {num_answers} diverse answers...")
            step2_start = time.time()
            answers = await self._generate_diverse_answers(
                requirement=requirement,
                research_goal=research_goal,
                collected_data=collected_data,
                context=context,
                num_answers=num_answers
            )
            step2_duration = time.time() - step2_start
            self.log(f"[REQ-GEN] ✓ Step 2 complete ({step2_duration:.2f}s)")
            self.log(f"[REQ-GEN] ✓ Generated {len(answers)} answers")

            # Step 3: Store answers in memory
            self.log("[REQ-GEN] Step 3/3: Storing answers in memory...")
            for i, answer in enumerate(answers, 1):
                self.memory.store_requirement_answer(answer)
                self.log(f"[REQ-GEN]    [{i}/{len(answers)}] Stored: {answer.id}")

            func_duration = time.time() - func_start
            self.log("=" * 60)
            self.log(f"[REQ-GEN] ✅ COMPLETE: {len(answers)} answers for {req_id}")
            self.log(f"[REQ-GEN] ⏱️  Total duration: {func_duration:.2f}s")
            self.log("=" * 60)

            return {
                "status": "success",
                "requirement_id": req_id,
                "answers": answers,
                "data_sources": {
                    "sources": list(collected_data.get("sources", {}).keys()),
                    "analysis": list(collected_data.get("analysis", {}).keys())
                }
            }

        except Exception as e:
            self.log(f"[REQ-GEN] ✗ Error: {e}", "error")
            import traceback
            self.log(f"[REQ-GEN] Traceback: {traceback.format_exc()}", "debug")
            return {
                "status": "error",
                "requirement_id": req_id,
                "message": str(e),
                "answers": []
            }

    async def _collect_data_for_requirement(
        self,
        requirement: Dict[str, Any],
        research_goal: ResearchGoal,
        context: Dict[str, RequirementAnswer]
    ) -> Dict[str, Any]:
        """
        Collect data relevant to a specific requirement using Hybrid 3-Stage System (NEW).

        NEW ARCHITECTURE (Problem-Agnostic):
        - Stage 1: Requirement Analysis - Extract entities from requirement + context
        - Stage 2: Unified Collection-Analysis - Execute tools based on entities
        - Stage 3: Answer Synthesis - Generate diverse RequirementAnswers (in _generate_diverse_answers)

        This method implements Stages 1 and 2.

        Changes from old version:
        1. No longer extracts target_name from research_goal (hardcoded protein extraction removed)
        2. Uses requirement + context for entity extraction (Stage 1)
        3. Uses entity-based tool selection (Stage 2)

        Args:
            requirement: The requirement to collect data for
            research_goal: Overall research goal (fallback context only)
            context: Confirmed answers from dependencies

        Returns:
            Collected data structure with entity_analysis, sources, and analysis
        """
        import time
        func_start = time.time()
        self.log("  ┌─ FUNCTION: _collect_data_for_requirement() [HYBRID 3-STAGE]")

        # Extract requirement details
        if isinstance(requirement, dict):
            req_id = requirement.get("requirement_id", requirement.get("step_id", ""))
            req_title = requirement.get("title", "")
            req_description = requirement.get("description", "")
            req_type = requirement.get("requirement_type", "answer")
            expected_deliverables = requirement.get("expected_deliverables", [])
        else:
            req_id = getattr(requirement, "requirement_id", "")
            req_title = getattr(requirement, "title", "")
            req_description = getattr(requirement, "description", "")
            req_type = getattr(requirement, "requirement_type", "answer")
            expected_deliverables = getattr(requirement, "expected_deliverables", [])

        self.log(f"  │ Requirement: {req_id} ({req_type})")
        self.log(f"  │ Expected deliverables: {len(expected_deliverables)}")
        self.log(f"  │ Context dependencies: {len(context)}")

        # Build context summary for logging
        context_summary = ""
        if context:
            context_parts = []
            for dep_id, answer in context.items():
                if isinstance(answer, dict):
                    ans_text = answer.get("answer", "")[:200]
                else:
                    ans_text = getattr(answer, "answer", "")[:200]
                context_parts.append(f"- {dep_id}: {ans_text}...")
            context_summary = "\n".join(context_parts)
            self.log(f"  │ Context preview: {len(context_summary)} chars")

        # ========== STAGE 1: Requirement Analysis (NEW!) ==========
        self.log("  │")
        self.log("  │ ┌─ STAGE 1: Requirement Analysis (Problem-Agnostic)")
        entity_analysis = await self._analyze_requirement_entities(
            requirement=requirement,
            context=context,
            research_goal=research_goal  # Fallback only
        )
        self.log("  │ └─ STAGE 1 COMPLETE")

        # ========== STAGE 2: Unified Collection-Analysis (REFACTORED) ==========
        self.log("  │")
        self.log("  │ ┌─ STAGE 2: Unified Collection + Analysis (Entity-Based)")
        collected_data = await self._execute_unified_collection_analysis(
            entity_analysis=entity_analysis,
            research_goal=research_goal,
            req_id=req_id  # NEW: Pass req_id for file saving
        )
        self.log("  │ └─ STAGE 2 COMPLETE")

        # Add requirement-specific metadata
        collected_data["requirement"] = {
            "id": req_id,
            "title": req_title,
            "description": req_description,
            "type": req_type,
            "expected_deliverables": expected_deliverables
        }
        collected_data["dependency_context"] = context_summary

        func_duration = time.time() - func_start
        self.log(f"  └─ FUNCTION COMPLETED ({func_duration:.2f}s)")
        self.log(f"     ✓ Entity analysis + data collection complete for {req_id}")

        return collected_data

    async def _analyze_requirement_entities(
        self,
        requirement: Dict[str, Any],
        context: Dict[str, RequirementAnswer],
        research_goal: ResearchGoal
    ) -> Dict[str, Any]:
        """
        Stage 1: Analyze requirement to extract entities and data needs (NEW - Problem-Agnostic).

        This replaces the hardcoded protein-centric target extraction with a flexible
        entity analysis that works for all biomedical research domains.

        Args:
            requirement: The requirement to analyze
            context: Confirmed answers from dependency requirements
            research_goal: Overall research goal (fallback context only)

        Returns:
            Entity analysis structure:
            {
                "primary_entities": [
                    {
                        "type": "protein|pathway|compound|disease|domain|gene",
                        "name": str,
                        "description": str,
                        "identifiers": {...},
                        "source": "requirement|context|research_goal",
                        "priority": "required|recommended|optional"
                    }
                ],
                "data_requirements": [
                    {
                        "type": "sequence|structure|pathway|literature|...",
                        "source": "uniprot|pdb|kegg|ncbi|...",
                        "priority": "required|recommended|optional",
                        "reason": str
                    }
                ],
                "analysis_needs": ["binding_energy", "docking", ...],
                "context_refinements": {
                    "binding_site": "CRD2-CRD3"  # Extracted from context.deliverables
                }
            }
        """
        import time
        func_start = time.time()
        self.log("  ┌─ FUNCTION: _analyze_requirement_entities() [STAGE 1 - NEW]")

        # Extract requirement details for logging
        req_id = requirement.get("requirement_id", requirement.get("step_id", "unknown"))
        req_title = requirement.get("title", "")
        self.log(f"  │ Requirement: {req_id} - {req_title}")
        self.log(f"  │ Context dependencies: {len(context)}")

        # Build prompt using the new template
        self.log("  │ [1/4] Preparing prompt from template...")
        try:
            prompt = self.prompt_manager.get_prompt(
                'generation/requirement_analysis',
                requirement=requirement,
                context=context,
                research_goal=research_goal
            )
            self.log(f"  │ ✓ Prompt prepared ({len(prompt)} chars)")
        except Exception as e:
            self.log(f"  │ ✗ Template rendering failed: {e}", "error")
            raise

        # Call LLM to extract entities
        try:
            self.log("  │ [2/4] Calling LLM.generate_json()...")
            self.log(f"  │   - temperature: 0.2 (low for consistent extraction)")
            self.log(f"  │   - purpose: requirement_entity_analysis")

            llm_start = time.time()
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent entity extraction
                purpose=f"requirement_entity_analysis_{req_id}"
            )
            llm_duration = time.time() - llm_start
            self.log(f"  │ ✓ LLM responded in {llm_duration:.2f}s")

            # Parse response structure
            self.log("  │ [3/4] Parsing entity analysis response...")
            entity_analysis = {
                "primary_entities": response.get("primary_entities", []),
                "data_requirements": response.get("data_requirements", []),
                "analysis_needs": response.get("analysis_needs", []),
                "context_refinements": response.get("context_refinements", {})
            }

            # Log summary
            num_entities = len(entity_analysis["primary_entities"])
            num_data_reqs = len(entity_analysis["data_requirements"])
            num_analysis = len(entity_analysis["analysis_needs"])
            self.log(f"  │ ✓ Extracted: {num_entities} entities, {num_data_reqs} data reqs, {num_analysis} analyses")

            # Detailed entity logging
            for entity in entity_analysis["primary_entities"]:
                entity_type = entity.get("type", "unknown")
                entity_name = entity.get("name", "unknown")
                entity_priority = entity.get("priority", "unknown")
                self.log(f"  │   - {entity_type}: {entity_name} ({entity_priority})")

            # Validation
            self.log("  │ [4/4] Validating entity analysis...")
            if num_entities == 0:
                self.log(f"  │ ⚠️  Warning: No entities extracted, using fallback", "warning")
                return self._create_fallback_entity_analysis(requirement, research_goal)

            # Structural validation
            if not self._validate_entity_analysis(entity_analysis):
                self.log(f"  │ ⚠️  Warning: Validation failed, using fallback", "warning")
                return self._create_fallback_entity_analysis(requirement, research_goal)

            self.log(f"  │ ✓ Validation passed")

            func_duration = time.time() - func_start
            self.log(f"  └─ FUNCTION COMPLETED: _analyze_requirement_entities()")
            self.log(f"     ⏱️  Duration: {func_duration:.2f}s")
            return entity_analysis

        except Exception as e:
            self.log(f"  │ ✗ Entity analysis failed: {e}", "error")
            import traceback
            self.log(f"  │ Traceback: {traceback.format_exc()}", "debug")
            self.log(f"  │ Using fallback entity analysis...")

            result = self._create_fallback_entity_analysis(requirement, research_goal)
            func_duration = time.time() - func_start
            self.log(f"  └─ FUNCTION COMPLETED (with fallback): _analyze_requirement_entities()")
            self.log(f"     ⏱️  Duration: {func_duration:.2f}s")
            return result

    def _create_fallback_entity_analysis(
        self,
        requirement: Dict[str, Any],
        research_goal: ResearchGoal
    ) -> Dict[str, Any]:
        """
        Create minimal entity analysis when Stage 1 fails.

        This provides graceful degradation - uses research_goal as fallback
        to maintain backward compatibility with old configs.
        """
        self.log("  │ Creating fallback entity analysis from research_goal...")

        # Extract requirement description as entity
        req_description = requirement.get("description", "")

        # Minimal entity structure
        fallback_entity = {
            "type": "unknown",
            "name": research_goal.description[:100],  # Truncate for safety
            "description": "Fallback entity from research goal",
            "identifiers": {},
            "source": "research_goal",
            "priority": "required"
        }

        # Minimal entity analysis
        entity_analysis = {
            "primary_entities": [fallback_entity],
            "data_requirements": [
                {
                    "type": "all",
                    "source": "all",
                    "priority": "recommended",
                    "reason": "Fallback: collect general data"
                }
            ],
            "analysis_needs": [],
            "context_refinements": {}
        }

        self.log(f"  │ ✓ Fallback entity analysis created")
        return entity_analysis

    def _validate_entity_analysis(self, entity_analysis: Dict[str, Any]) -> bool:
        """
        Validate entity analysis structure and content (Phase 1.3).

        Checks:
        1. All required keys present
        2. Entity types are valid
        3. Priorities are valid
        4. Data requirement structure is valid

        Returns:
            True if valid, False otherwise
        """
        # Valid entity types (problem-agnostic)
        VALID_ENTITY_TYPES = {"protein", "gene", "pathway", "compound", "disease", "domain", "organism", "assay"}
        VALID_PRIORITIES = {"required", "recommended", "optional"}
        VALID_DATA_TYPES = {"sequence", "structure", "pathway", "interaction",
                           "literature", "expression", "variant", "binding_affinity",
                           "mutation_effect", "taxonomy", "all"}

        try:
            # Check required keys
            if not all(key in entity_analysis for key in
                      ["primary_entities", "data_requirements", "analysis_needs", "context_refinements"]):
                self.log("  │ ✗ Validation failed: missing required keys", "warning")
                return False

            # Validate primary entities
            for entity in entity_analysis["primary_entities"]:
                # Check required fields
                if not all(key in entity for key in ["type", "name", "priority"]):
                    self.log(f"  │ ✗ Entity missing required fields: {entity}", "warning")
                    return False

                # Check entity type
                if entity["type"] not in VALID_ENTITY_TYPES and entity["type"] != "unknown":
                    self.log(f"  │ ⚠️  Warning: Unknown entity type: {entity['type']}", "warning")
                    # Don't fail - allow LLM flexibility

                # Check priority
                if entity["priority"] not in VALID_PRIORITIES:
                    self.log(f"  │ ✗ Invalid priority: {entity['priority']}", "warning")
                    return False

            # Validate data requirements
            for data_req in entity_analysis["data_requirements"]:
                # Check required fields
                if not all(key in data_req for key in ["type", "source", "priority"]):
                    self.log(f"  │ ✗ Data requirement missing fields: {data_req}", "warning")
                    return False

                # Check data type
                if data_req["type"] not in VALID_DATA_TYPES:
                    self.log(f"  │ ⚠️  Warning: Unknown data type: {data_req['type']}", "warning")
                    # Don't fail - allow flexibility

                # Check priority
                if data_req["priority"] not in VALID_PRIORITIES:
                    self.log(f"  │ ✗ Invalid priority: {data_req['priority']}", "warning")
                    return False

            # Validate analysis needs (should be list of strings)
            if not isinstance(entity_analysis["analysis_needs"], list):
                self.log("  │ ✗ analysis_needs must be a list", "warning")
                return False

            for analysis in entity_analysis["analysis_needs"]:
                if not isinstance(analysis, str):
                    self.log(f"  │ ✗ Invalid analysis need (not string): {analysis}", "warning")
                    return False

            # Validate context refinements (should be dict)
            if not isinstance(entity_analysis["context_refinements"], dict):
                self.log("  │ ✗ context_refinements must be a dict", "warning")
                return False

            # All checks passed
            return True

        except Exception as e:
            self.log(f"  │ ✗ Validation exception: {e}", "error")
            return False

    async def _generate_diverse_answers(
        self,
        requirement: Dict[str, Any],
        research_goal: ResearchGoal,
        collected_data: Dict[str, Any],
        context: Dict[str, RequirementAnswer],
        num_answers: int = 3
    ) -> List[RequirementAnswer]:
        """
        Generate N diverse answers for a requirement.

        Diversity strategies:
        1. Conservative: Safe, well-established approach
        2. Moderate: Balanced innovation and safety
        3. Aggressive: Novel, higher-risk approach

        Args:
            requirement: The requirement to answer
            research_goal: Overall research goal
            collected_data: Data from 2-Chain collection
            context: Confirmed answers from dependencies
            num_answers: Number of diverse answers to generate

        Returns:
            List of RequirementAnswer objects
        """
        import time
        func_start = time.time()
        self.log("  ┌─ FUNCTION: _generate_diverse_answers()")

        # Extract requirement details
        if isinstance(requirement, dict):
            req_id = requirement.get("requirement_id", requirement.get("step_id", ""))
            req_title = requirement.get("title", "")
            req_description = requirement.get("description", "")
            req_type = requirement.get("requirement_type", "answer")
            expected_deliverables = requirement.get("expected_deliverables", [])
            depends_on = requirement.get("depends_on", [])
        else:
            req_id = getattr(requirement, "requirement_id", "")
            req_title = getattr(requirement, "title", "")
            req_description = getattr(requirement, "description", "")
            req_type = getattr(requirement, "requirement_type", "answer")
            expected_deliverables = getattr(requirement, "expected_deliverables", [])
            depends_on = getattr(requirement, "depends_on", [])

        # Prepare context for prompt
        parsed_problem = self.config.get("parsed_problem", {})

        # Convert context (RequirementAnswer objects) to serializable format
        context_for_prompt = {}
        for dep_id, answer in context.items():
            if isinstance(answer, dict):
                context_for_prompt[dep_id] = answer
            else:
                context_for_prompt[dep_id] = {
                    "requirement_id": answer.requirement_id,
                    "requirement_title": answer.requirement_title,
                    "answer": answer.answer,
                    "rationale": answer.rationale,
                    "deliverables": answer.deliverables,
                    "confidence": answer.confidence
                }

        # === MODIFIED: File-based mode with fallback ===
        if self.data_file_manager and "_file_paths" in collected_data:
            self.log(f"  │ Using file-based analysis results (zero truncation)")

            # Load ONLY Chain 2 analysis results (not Chain 1 data)
            analysis_results = self.data_file_manager.load_analysis_results(req_id)

            truncated_data = {
                "sources": {},  # Empty - not needed for Answer Gen
                "analysis": analysis_results,  # Full analysis results
                "_file_paths": collected_data["_file_paths"],
                "_metadata_only": False  # Analysis results are complete
            }
        else:
            # FALLBACK: Old truncation (backward compatible)
            self.log(f"  │ Using legacy truncation (no DataFileManager)")
            truncated_data = self._truncate_data_for_prompt(collected_data)

        # Get static constraints
        static_constraints = self._extract_static_constraints()

        # Build prompt for diverse answer generation
        self.log(f"  │ Building prompt for {num_answers} diverse answers...")
        prompt = self.prompt_manager.get_prompt(
            "generation/diverse_answers",
            parsed_problem=parsed_problem,
            research_goal=research_goal,
            requirement={
                "requirement_id": req_id,
                "title": req_title,
                "description": req_description,
                "requirement_type": req_type,
                "expected_deliverables": expected_deliverables,
                "depends_on": depends_on
            },
            context=context_for_prompt,
            data_sources={},  # Empty - Chain 1 data not needed
            analysis_results=truncated_data.get("analysis", {}),  # Full results
            constraints=static_constraints.get("hard", []),
            num_answers=num_answers
        )
        self.log(f"  │ Prompt prepared: {len(prompt)} chars")

        try:
            self.log(f"  │ Calling LLM.generate_json()...")
            llm_start = time.time()
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # Higher for diversity
                purpose=f"diverse_answers_{req_id}"
            )
            llm_duration = time.time() - llm_start
            self.log(f"  │ ✓ LLM responded in {llm_duration:.2f}s")

            # Parse response into RequirementAnswer objects
            answers = []
            raw_answers = response.get("answers", [])
            self.log(f"  │ Parsing {len(raw_answers)} answers...")

            for i, ans_data in enumerate(raw_answers):
                answer_id = f"ra_{req_id}_{uuid.uuid4().hex[:8]}"

                # Determine approach/strategy for this answer
                approach = ans_data.get("approach", f"approach_{i+1}")

                # Extract tool usage info
                tool_usage = collected_data.get("tool_usage", {})

                # Build comprehensive metadata with entity-based tracking
                metadata = {
                    "approach": approach,
                    "strategy": ans_data.get("strategy", ""),
                    "innovation_level": ans_data.get("innovation_level", "moderate"),

                    # === NEW: File paths ===
                    "data_files": collected_data.get("_file_paths", {}),

                    # Entity analysis (NEW!)
                    "entity_analysis": collected_data.get("entity_analysis", {}),

                    # Data collection tracking
                    "data_collection": {
                        "servers_used": list(collected_data.get("sources", {}).keys()),
                        "tools": tool_usage.get("collection_tools", []),
                        "sources_file": collected_data.get("_file_paths", {}).get("sources_file"),  # NEW
                        # REMOVED: "sources_detail" (now in file)
                    },

                    # Data analysis tracking
                    "data_analysis": {
                        "analyses_performed": collected_data.get("entity_analysis", {}).get("analysis_needs", []),
                        "tools": tool_usage.get("analysis_tools", []),
                        "results_file": collected_data.get("_file_paths", {}).get("results_file"),  # NEW
                        # REMOVED: "results" (now in file)
                    }
                }

                answer = RequirementAnswer(
                    id=answer_id,
                    requirement_id=req_id,
                    requirement_title=req_title,
                    answer=ans_data.get("answer", ""),
                    rationale=ans_data.get("rationale", ""),
                    deliverables=ans_data.get("deliverables", {}),
                    confidence=max(0.0, min(1.0, float(ans_data.get("confidence", 0.5)))),
                    builds_on=depends_on,
                    status="generated",
                    elo_rating=1200.0,
                    generated_at=datetime.now(),
                    generation_method="data_based_diverse",
                    data_sources=list(collected_data.get("sources", {}).keys()),
                    metadata=metadata
                )
                answers.append(answer)
                self.log(f"  │    [{i+1}/{len(raw_answers)}] Created: {answer_id} ({approach})")

            func_duration = time.time() - func_start
            self.log(f"  └─ FUNCTION COMPLETED ({func_duration:.2f}s) - {len(answers)} answers")

            return answers

        except Exception as e:
            self.log(f"  │ ✗ Error generating diverse answers: {e}", "error")
            import traceback
            self.log(f"  │ {traceback.format_exc()}", "debug")
            self.log(f"  └─ FUNCTION FAILED")
            return []

    async def generate_evolved_answer(
        self,
        parent_answer: RequirementAnswer,
        requirement: Dict[str, Any],
        context: Dict[str, RequirementAnswer],
        evolution_method: str = "grounding"
    ) -> Optional[RequirementAnswer]:
        """
        Generate an evolved version of an existing answer.

        This supports the Evolution phase of Sequential Confirmation.

        Args:
            parent_answer: The answer to evolve from
            requirement: The requirement being answered
            context: Confirmed answers from dependencies
            evolution_method: Evolution strategy (grounding, coherence, simplification, divergent)

        Returns:
            New RequirementAnswer that is an evolved version of parent
        """
        import time
        func_start = time.time()

        req_id = parent_answer.requirement_id
        self.log(f"[EVOLVE] Evolving answer {parent_answer.id} using '{evolution_method}'")

        # Build prompt for evolution
        parsed_problem = self.config.get("parsed_problem", {})

        # Convert parent answer to dict for prompt
        parent_dict = {
            "id": parent_answer.id,
            "answer": parent_answer.answer,
            "rationale": parent_answer.rationale,
            "deliverables": parent_answer.deliverables,
            "confidence": parent_answer.confidence,
            "review": parent_answer.review
        }

        # Convert context
        context_for_prompt = {}
        for dep_id, answer in context.items():
            if isinstance(answer, dict):
                context_for_prompt[dep_id] = answer
            else:
                context_for_prompt[dep_id] = {
                    "requirement_id": answer.requirement_id,
                    "answer": answer.answer,
                    "deliverables": answer.deliverables
                }

        prompt = self.prompt_manager.get_prompt(
            "generation/answer_evolution",
            parsed_problem=parsed_problem,
            requirement=requirement,
            parent_answer=parent_dict,
            context=context_for_prompt,
            evolution_method=evolution_method
        )

        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                purpose=f"evolve_answer_{req_id}"
            )

            answer_id = f"ra_{req_id}_{uuid.uuid4().hex[:8]}"

            # Inherit parent's metadata (entity_analysis, data_collection, data_analysis)
            parent_metadata = getattr(parent_answer, "metadata", {}) or {}

            # Build evolved answer metadata - inherit from parent
            evolved_metadata = {
                "parent_id": parent_answer.id,
                "evolution_method": evolution_method,
                "improvements": response.get("improvements", []),

                # Inherit approach/strategy from parent
                "approach": parent_metadata.get("approach", ""),
                "strategy": parent_metadata.get("strategy", ""),
                "innovation_level": parent_metadata.get("innovation_level", ""),

                # Inherit entity analysis (no new data collection)
                "entity_analysis": parent_metadata.get("entity_analysis", {}),

                # Inherit data collection info (evolution doesn't collect new data)
                "data_collection": parent_metadata.get("data_collection", {}),

                # Inherit data analysis info
                "data_analysis": parent_metadata.get("data_analysis", {})
            }

            evolved = RequirementAnswer(
                id=answer_id,
                requirement_id=req_id,
                requirement_title=parent_answer.requirement_title,
                answer=response.get("answer", ""),
                rationale=response.get("rationale", ""),
                deliverables=response.get("deliverables", {}),
                confidence=max(0.0, min(1.0, float(response.get("confidence", 0.5)))),
                builds_on=parent_answer.builds_on,
                status="generated",
                elo_rating=1200.0,
                parent_ids=[parent_answer.id],
                evolution_method=evolution_method,
                iteration=parent_answer.iteration + 1,
                generated_at=datetime.now(),
                generation_method="evolution",
                data_sources=parent_answer.data_sources,  # Inherit data sources
                metadata=evolved_metadata
            )

            func_duration = time.time() - func_start
            self.log(f"[EVOLVE] ✓ Created evolved answer {answer_id} ({func_duration:.2f}s)")

            return evolved

        except Exception as e:
            self.log(f"[EVOLVE] ✗ Failed to evolve answer: {e}", "error")
            return None

    # ========================================================================
    # END: RequirementAnswer-based Generation
    # ========================================================================
    # NOTE: Old protein-centric methods (_extract_target_name, _extract_target_fallback)
    # have been DELETED in Phase 3.2 of the Problem-Agnostic refactoring.
    # Replaced by _analyze_requirement_entities() which supports all biomedical domains.
    # ========================================================================

    async def _execute_unified_collection_analysis(
        self,
        entity_analysis: Dict[str, Any],
        research_goal: ResearchGoal,
        req_id: str  # NEW: Required for file saving
    ) -> Dict[str, Any]:
        """
        Stage 2: Execute Unified Collection + Analysis (NEW - Problem-Agnostic).

        Refactored from _collect_multi_tool_data() to use entity_analysis instead of target_name.

        Changes from old 2-Chain:
        1. Parameter: entity_analysis (from Stage 1) instead of target_name
        2. Tool filtering: Based on entity types instead of problem_type
        3. Prompt: Passes entity_analysis to LLM instead of target_name

        Args:
            entity_analysis: Entity analysis from Stage 1 with:
                - primary_entities: Detected entities (protein, pathway, etc.)
                - data_requirements: What data sources are needed
                - analysis_needs: What analyses to perform
                - context_refinements: Refinements from previous requirements
            research_goal: Overall research goal (for additional context)

        Returns:
            Collected data structure with sources and analysis results
        """
        import time
        func_start = time.time()
        self.log("─" * 60)
        self.log("🔧 FUNCTION: _execute_unified_collection_analysis() [STAGE 2 - NEW]")
        self.log("─" * 60)

        # Extract entity summary for logging
        num_entities = len(entity_analysis.get("primary_entities", []))
        num_data_reqs = len(entity_analysis.get("data_requirements", []))
        num_analyses = len(entity_analysis.get("analysis_needs", []))
        self.log(f"[STAGE-2] Entity Analysis Summary:")
        self.log(f"[STAGE-2]   - Entities: {num_entities}")
        self.log(f"[STAGE-2]   - Data requirements: {num_data_reqs}")
        self.log(f"[STAGE-2]   - Analysis needs: {num_analyses}")

        # Step 1: Extract entity types for tool filtering (NEW - replaces problem_type)
        self.log("[STAGE-2] Step 1: Extracting entity types for tool filtering...")
        entity_types = set()
        for entity in entity_analysis.get("primary_entities", []):
            entity_type = entity.get("type", "unknown")
            entity_types.add(entity_type)
        self.log(f"[STAGE-2] ✓ Entity types: {list(entity_types)}")

        # Step 2: Initialize collected data structure
        self.log("[STAGE-2] Step 2: Initializing data structure...")
        collected_data = {
            "entity_analysis": entity_analysis,  # NEW: Include entity analysis
            "research_goal": research_goal.description,
            "sources": {},
            "analysis": {},
            # Tool usage tracking for config export
            "tool_usage": {
                "collection_tools": [],  # Chain 1 tools
                "analysis_tools": []     # Chain 2 tools
            }
        }
        self.log("[STAGE-2] ✓ Data structure initialized")

        # ========== Chain 1: 데이터 수집 ==========
        self.log("=" * 60)
        self.log("[CHAIN-1] 🔗 STARTING DATA COLLECTION CHAIN (Entity-Based)")
        self.log("=" * 60)
        chain1_start = time.time()

        self.log("[CHAIN-1] [1/5] Getting collection tools from registry...")
        # NEW: Get all collection tools (entity-agnostic filtering done by LLM)
        collection_tools = self.tool_registry.get_collection_tools(
            problem_type="all",  # Get all tools, LLM will filter based on entity_analysis
            stage="generation"
        )
        self.log(f"[CHAIN-1] ✓ Found {len(collection_tools)} collection tools (all types)")

        if collection_tools:
            self.log(f"[CHAIN-1]    Sample tools: {[t['function']['name'] for t in collection_tools[:5]]}{'...' if len(collection_tools) > 5 else ''}")

            # LLM tool calling prompt for Chain 1 (NEW: uses entity_analysis)
            self.log("[CHAIN-1] [2/5] Preparing prompt for LLM tool selection...")
            tool_names = ', '.join([t['function']['name'] for t in collection_tools])
            collection_prompt = self.prompt_manager.get_prompt(
                'generation/chain1_collection',
                research_goal=research_goal,
                entity_analysis=entity_analysis,  # NEW: Pass entity_analysis instead of target_name
                tool_names=tool_names,
                context_refinements=entity_analysis.get("context_refinements", {})  # NEW
            )
            self.log(f"[CHAIN-1] ✓ Prompt prepared ({len(collection_prompt)} chars)")
            self.log(f"[CHAIN-1]    Entity-driven tool selection enabled")

            try:
                self.log("[CHAIN-1] [3/5] Calling LLM.generate_with_tools()...")
                self.log(f"[CHAIN-1]    - temperature: 0.1")
                self.log(f"[CHAIN-1]    - max_iterations: 5")

                llm_start = time.time()
                collection_result = await self.llm.generate_with_tools(
                    messages=[{"role": "user", "content": collection_prompt}],
                    tools=collection_tools,
                    tool_choice="auto",
                    temperature=0.1,
                    max_iterations=5,
                    purpose="data_collection"
                )
                llm_duration = time.time() - llm_start

                tool_calls = collection_result.get("tool_calls", [])
                self.log(f"[CHAIN-1] ✓ LLM responded in {llm_duration:.2f}s")
                self.log(f"[CHAIN-1] ✓ LLM selected {len(tool_calls)} tool(s) to execute")

                # Execute tool calls via ToolExecutor
                self.log("[CHAIN-1] [4/5] Executing selected tools...")
                for idx, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})

                    try:
                        self.log(f"[CHAIN-1]    [{idx}/{len(tool_calls)}] Starting: {tool_name}")
                        tool_start = time.time()
                        result = await self.tool_executor.execute_tool(tool_name, arguments)
                        tool_duration = time.time() - tool_start

                        # Store results by server name
                        server_name = self._get_server_name_from_tool(tool_name)
                        if server_name not in collected_data["sources"]:
                            collected_data["sources"][server_name] = []
                        collected_data["sources"][server_name].append({
                            "tool": tool_name,
                            "result": result
                        })

                        # Track tool usage for config export
                        collected_data["tool_usage"]["collection_tools"].append({
                            "tool_name": tool_name,
                            "server": server_name,
                            "arguments": arguments,
                            "duration_seconds": round(tool_duration, 2),
                            "status": "success"
                        })

                        self.log(f"[CHAIN-1]    [{idx}/{len(tool_calls)}] ✓ {tool_name} completed in {tool_duration:.2f}s")
                        self.log(f"[CHAIN-1]       Stored in: {server_name}")

                    except Exception as e:
                        self.log(f"[CHAIN-1]    [{idx}/{len(tool_calls)}] ✗ {tool_name} failed: {e}", "warning")
                        # Track failed tool usage
                        collected_data["tool_usage"]["collection_tools"].append({
                            "tool_name": tool_name,
                            "server": self._get_server_name_from_tool(tool_name),
                            "arguments": arguments,
                            "status": "failed",
                            "error": str(e)
                        })

                self.log("[CHAIN-1] [5/5] Tool execution complete")

            except Exception as e:
                self.log(f"[CHAIN-1] ✗ Chain 1 error: {e}", "error")

        chain1_duration = time.time() - chain1_start
        self.log("=" * 60)
        self.log(f"[CHAIN-1] ✅ CHAIN 1 COMPLETE")
        self.log(f"[CHAIN-1] ⏱️  Duration: {chain1_duration:.2f}s")
        self.log(f"[CHAIN-1] 📊 Collected from {len(collected_data['sources'])} source(s): {list(collected_data['sources'].keys())}")
        self.log("=" * 60)

        # === NEW: Save Chain 1 data to files ===
        if self.data_file_manager:
            self.log("[CHAIN-1] 💾 Saving data to files...")

            try:
                file_paths = self.data_file_manager.save_collection_data(req_id, collected_data)
                collected_data["_file_paths"] = file_paths  # Store for downstream use

                self.log(f"[CHAIN-1] ✓ Full data saved: {file_paths['sources_file']}")
                self.log(f"[CHAIN-1] ✓ Metadata saved: {file_paths['metadata_file']}")

                # Log file size
                import os
                sources_size = os.path.getsize(file_paths['sources_file'])
                self.log(f"[CHAIN-1] 📦 Size: {sources_size / 1024:.1f} KB (no truncation)")

            except Exception as e:
                self.log(f"[CHAIN-1] ⚠️  Failed to save files: {e}", "warning")
                # Continue without files (fallback to old truncation)

        # ========== Chain 2: 데이터 분석 ==========
        self.log("=" * 60)
        self.log("[CHAIN-2] 🔬 STARTING DATA ANALYSIS CHAIN (Entity-Based)")
        self.log("=" * 60)
        chain2_start = time.time()

        self.log("[CHAIN-2] [1/5] Getting analysis tools from registry...")
        # NEW: Get all analysis tools (entity-agnostic filtering done by LLM)
        analysis_tools = self.tool_registry.get_analysis_tools(
            problem_type="all",  # Get all tools, LLM will filter based on analysis_needs
            stage="generation"
        )
        self.log(f"[CHAIN-2] ✓ Found {len(analysis_tools)} analysis tools (all types)")

        if not analysis_tools:
            self.log("[CHAIN-2] ⚠ No analysis tools available, skipping Chain 2")
            func_duration = time.time() - func_start
            self.log("─" * 60)
            self.log(f"🔧 FUNCTION COMPLETED: _execute_unified_collection_analysis()")
            self.log(f"⏱️  Total duration: {func_duration:.2f}s")
            self.log(f"📊 Final result: {len(collected_data['sources'])} sources, {len(collected_data['analysis'])} analyses")
            self.log("─" * 60)
            return collected_data

        self.log(f"[CHAIN-2]    Sample tools: {[t['function']['name'] for t in analysis_tools[:5]]}{'...' if len(analysis_tools) > 5 else ''}")

        # Prepare collected data summary for LLM
        self.log("[CHAIN-2] [2/5] Summarizing Chain 1 data for LLM...")
        data_summary = self._summarize_collected_data(collected_data)
        self.log(f"[CHAIN-2] ✓ Summary prepared ({len(data_summary)} chars)")

        # LLM tool calling prompt for Chain 2 (NEW: uses entity_analysis)
        self.log("[CHAIN-2] [3/5] Preparing prompt for LLM tool selection...")
        analysis_tool_names = ', '.join([t['function']['name'] for t in analysis_tools])
        analysis_prompt = self.prompt_manager.get_prompt(
            'generation/chain2_analysis',
            research_goal=research_goal,
            entity_analysis=entity_analysis,  # NEW: Pass entity_analysis
            analysis_needs=entity_analysis.get("analysis_needs", []),  # NEW
            collected_data_summary=data_summary,
            tool_names=analysis_tool_names,
            context_refinements=entity_analysis.get("context_refinements", {})  # NEW
        )
        self.log(f"[CHAIN-2] ✓ Prompt prepared ({len(analysis_prompt)} chars)")
        self.log(f"[CHAIN-2]    Analysis needs: {entity_analysis.get('analysis_needs', [])}")

        try:
            self.log("[CHAIN-2] [4/5] Calling LLM.generate_with_tools()...")
            self.log(f"[CHAIN-2]    - temperature: 0.1")
            self.log(f"[CHAIN-2]    - max_iterations: 5")

            llm_start = time.time()
            analysis_result = await self.llm.generate_with_tools(
                messages=[{"role": "user", "content": analysis_prompt}],
                tools=analysis_tools,
                tool_choice="auto",
                temperature=0.1,
                max_iterations=5,
                purpose="data_analysis"
            )
            llm_duration = time.time() - llm_start

            tool_calls = analysis_result.get("tool_calls", [])
            self.log(f"[CHAIN-2] ✓ LLM responded in {llm_duration:.2f}s")
            self.log(f"[CHAIN-2] ✓ LLM selected {len(tool_calls)} tool(s) to execute")

            # Execute analysis tool calls
            self.log("[CHAIN-2] [5/5] Executing analysis tools...")
            for idx, tool_call in enumerate(tool_calls, 1):
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})

                try:
                    self.log(f"[CHAIN-2]    [{idx}/{len(tool_calls)}] Starting: {tool_name}")
                    tool_start = time.time()
                    result = await self.tool_executor.execute_tool(tool_name, arguments)
                    tool_duration = time.time() - tool_start

                    collected_data["analysis"][tool_name] = result

                    # Track tool usage for config export
                    collected_data["tool_usage"]["analysis_tools"].append({
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "duration_seconds": round(tool_duration, 2),
                        "status": "success"
                    })

                    self.log(f"[CHAIN-2]    [{idx}/{len(tool_calls)}] ✓ {tool_name} completed in {tool_duration:.2f}s")

                except Exception as e:
                    self.log(f"[CHAIN-2]    [{idx}/{len(tool_calls)}] ✗ {tool_name} failed: {e}", "warning")
                    # Track failed tool usage
                    collected_data["tool_usage"]["analysis_tools"].append({
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "status": "failed",
                        "error": str(e)
                    })

        except Exception as e:
            self.log(f"[CHAIN-2] ✗ Chain 2 error: {e}", "error")

        chain2_duration = time.time() - chain2_start
        self.log("=" * 60)
        self.log(f"[CHAIN-2] ✅ CHAIN 2 COMPLETE")
        self.log(f"[CHAIN-2] ⏱️  Duration: {chain2_duration:.2f}s")
        self.log(f"[CHAIN-2] 📊 Analysis results: {len(collected_data['analysis'])} tool(s) - {list(collected_data['analysis'].keys())}")
        self.log("=" * 60)

        # === NEW: Save Chain 2 analysis results ===
        if self.data_file_manager and collected_data.get("analysis"):
            self.log("[CHAIN-2] 💾 Saving analysis results to files...")

            try:
                analysis_paths = self.data_file_manager.save_analysis_data(
                    req_id,
                    collected_data["analysis"]
                )
                collected_data["_file_paths"]["results_file"] = analysis_paths["results_file"]

                self.log(f"[CHAIN-2] ✓ Analysis saved: {analysis_paths['results_file']}")
            except Exception as e:
                self.log(f"[CHAIN-2] ⚠️  Failed to save analysis: {e}", "warning")

        # Function exit
        func_duration = time.time() - func_start
        self.log("─" * 60)
        self.log(f"🔧 FUNCTION COMPLETED: _execute_unified_collection_analysis() [STAGE 2]")
        self.log(f"⏱️  Total duration: {func_duration:.2f}s ({func_duration/60:.1f} min)")
        self.log(f"   - Chain 1 (Collection): {chain1_duration:.2f}s")
        self.log(f"   - Chain 2 (Analysis): {chain2_duration:.2f}s")
        self.log(f"📊 Final result:")
        self.log(f"   - Entity types processed: {list(entity_types)}")
        self.log(f"   - Sources: {len(collected_data['sources'])} - {list(collected_data['sources'].keys())}")
        self.log(f"   - Analyses: {len(collected_data['analysis'])} - {list(collected_data['analysis'].keys())}")
        self.log("─" * 60)

        return collected_data


    def _get_server_name_from_tool(self, tool_name: str) -> str:
        """Get server name from tool name"""
        tool_lower = tool_name.lower()

        # Map common tool name patterns to server names
        server_mappings = {
            # KEGG tools
            "kegg": "KEGG",
            "pathway": "KEGG",
            "disease": "KEGG",
            "drug": "KEGG",
            "compound": "KEGG",
            "enzyme": "KEGG",
            "reaction": "KEGG",
            "module": "KEGG",
            "brite": "KEGG",
            "ortholog": "KEGG",
            # UniProt tools
            "uniprot": "UniProt",
            "protein": "UniProt",  # search_proteins, get_protein_info
            "gene": "UniProt",  # search_by_gene
            "sequence": "UniProt",
            "feature": "UniProt",
            "domain": "UniProt",
            "variant": "UniProt",
            "homolog": "UniProt",
            "taxonomy": "UniProt",
            # NCBI tools
            "ncbi": "NCBI",
            "pubmed": "NCBI",
            "article": "NCBI",
            "mesh": "NCBI",
            # Other tools
            "scholar": "Scholar",
            "reactome": "Reactome",
            "pdb": "PDB",
            "structure": "PDB",
            "rosetta": "Rosetta",
            "pyrosetta": "Rosetta",
            "pymol": "PyMOL",
            "parse_and_execute": "PyMOL",
            "esmfold": "ESMFold",
            "fold": "ESMFold",
            "blast": "BLAST",
        }

        for pattern, server in server_mappings.items():
            if pattern in tool_lower:
                return server

        return tool_name  # Return tool name if no mapping found

    def _summarize_collected_data(self, collected_data: Dict[str, Any]) -> str:
        """Summarize collected data for Chain 2 prompt"""
        summary_parts = []

        for source, data in collected_data.get("sources", {}).items():
            if isinstance(data, list):
                # Multiple results from same source
                summary_parts.append(f"### {source}\n")
                for item in data[:3]:  # Limit to 3 items per source
                    tool_name = item.get("tool", "unknown")
                    result = item.get("result") or {}  # Handle None result
                    try:
                        result_str = json.dumps(result, indent=2, default=str)[:1000]
                    except (TypeError, ValueError):
                        result_str = str(result)[:1000]
                    summary_parts.append(f"**{tool_name}**:\n```json\n{result_str}\n```\n")
            else:
                try:
                    result_str = json.dumps(data or {}, indent=2, default=str)[:1000]
                except (TypeError, ValueError):
                    result_str = str(data)[:1000]
                summary_parts.append(f"### {source}\n```json\n{result_str}\n```\n")

        return "\n".join(summary_parts) if summary_parts else "No data collected."

    def _truncate_data_for_prompt(
        self,
        collected_data: Dict[str, Any],
        max_chars_per_source: int = 5000,
        max_total_chars: int = 50000
    ) -> Dict[str, Any]:
        """
        Truncate collected data to fit within LLM context limits.

        Args:
            collected_data: Raw collected data from Chain 1 & 2
            max_chars_per_source: Max characters per data source
            max_total_chars: Max total characters for all data

        Returns:
            Truncated data structure suitable for prompt
        """
        truncated = {
            "target": collected_data.get("target", ""),
            "research_goal": collected_data.get("research_goal", "")[:500],
            "sources": {},
            "analysis": {}
        }

        total_chars = 0

        # Truncate sources (Chain 1 data)
        for source, data in collected_data.get("sources", {}).items():
            if total_chars >= max_total_chars:
                truncated["sources"][source] = "[TRUNCATED: Context limit reached]"
                continue

            if isinstance(data, list):
                # Multiple results from same source - take first 2 items
                truncated_items = []
                for item in data[:2]:
                    tool_name = item.get("tool", "unknown")
                    result = item.get("result", {})
                    # Smart truncation: extract key fields only
                    result_summary = self._extract_key_fields(result, max_chars_per_source // 2)
                    truncated_items.append({
                        "tool": tool_name,
                        "result": result_summary
                    })
                truncated["sources"][source] = truncated_items
                total_chars += len(json.dumps(truncated_items, default=str))
            else:
                result_summary = self._extract_key_fields(data, max_chars_per_source)
                truncated["sources"][source] = result_summary
                total_chars += len(json.dumps(result_summary, default=str))

        # Truncate analysis results (Chain 2 data)
        for tool_name, result in collected_data.get("analysis", {}).items():
            if total_chars >= max_total_chars:
                truncated["analysis"][tool_name] = "[TRUNCATED: Context limit reached]"
                continue

            result_summary = self._extract_key_fields(result, max_chars_per_source)
            truncated["analysis"][tool_name] = result_summary
            total_chars += len(json.dumps(result_summary, default=str))

        self.log(f"Truncated data: {total_chars} chars (limit: {max_total_chars})")
        self.log(f"  Sources: {list(truncated['sources'].keys())}, Analysis: {list(truncated['analysis'].keys())}")
        return truncated

    def _extract_key_fields(self, data: Any, max_chars: int = 2000) -> Any:
        """
        Extract key fields from data, prioritizing important biological information.

        For protein data: accession, name, sequence (truncated), features
        For pathway data: id, name, genes (first 10)
        For general data: truncate to max_chars
        """
        if data is None:
            return None

        if isinstance(data, str):
            return data[:max_chars] + "..." if len(data) > max_chars else data

        if isinstance(data, (int, float, bool)):
            return data

        if isinstance(data, list):
            # Truncate list to first 5 items
            truncated_list = []
            char_count = 0
            for item in data[:5]:
                item_str = json.dumps(item, default=str)
                if char_count + len(item_str) > max_chars:
                    break
                truncated_list.append(self._extract_key_fields(item, max_chars // 5))
                char_count += len(item_str)
            if len(data) > 5:
                truncated_list.append(f"... and {len(data) - 5} more items")
            return truncated_list

        if isinstance(data, dict):
            # Priority fields for biological data
            priority_keys = [
                # Protein fields
                "accession", "id", "name", "gene", "organism", "function",
                "sequence", "length", "features", "domains", "binding_sites",
                # Pathway fields
                "pathway_id", "pathway_name", "genes", "compounds",
                # Structure fields
                "pdb_id", "resolution", "method", "chains",
                # Analysis fields
                "score", "energy", "rmsd", "results", "summary"
            ]

            result = {}
            char_count = 0

            # First pass: priority keys
            for key in priority_keys:
                if key in data and char_count < max_chars:
                    value = data[key]
                    # Special handling for sequence (truncate to 100 chars)
                    if key == "sequence" and isinstance(value, str) and len(value) > 100:
                        value = value[:100] + f"... ({len(value)} total)"
                    result[key] = self._extract_key_fields(value, max_chars // 10)
                    char_count += len(json.dumps(result[key], default=str))

            # Second pass: remaining keys if space allows
            remaining_chars = max_chars - char_count
            for key, value in data.items():
                if key not in result and remaining_chars > 100:
                    truncated_value = self._extract_key_fields(value, remaining_chars // 5)
                    value_str = json.dumps(truncated_value, default=str)
                    if len(value_str) < remaining_chars:
                        result[key] = truncated_value
                        remaining_chars -= len(value_str)

            return result

        # Fallback: convert to string and truncate
        str_data = str(data)
        return str_data[:max_chars] + "..." if len(str_data) > max_chars else str_data
    
    def _extract_static_constraints(self) -> Dict[str, Any]:
        """
        Extract static constraints from research config.

        Supports both v3.0 (list) and v2.0 (dict) formats for migration period.

        Returns:
            Dict with "hard", "soft", and "pipeline_requirements" keys
        """
        constraints = self.config.get("constraints", {})

        # v3.0 NEW FORMAT: Simple list (preferred)
        if isinstance(constraints, list):
            return {
                "hard": constraints,
                "soft": [],
                "pipeline_requirements": []
            }

        # FALLBACK: Empty constraints
        return {"hard": [], "soft": [], "pipeline_requirements": []}

