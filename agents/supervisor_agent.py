"""
Supervisor Agent - Sequential Confirmation Research Orchestrator

This is the RequirementAnswer-based Supervisor that:
1. Uses ConfigurationAgent to parse research goals into Requirements
2. Processes Requirements in topological order (dependency-aware)
3. Generates diverse answers for each Requirement
4. Evaluates, ranks, and confirms answers before proceeding
5. Uses confirmed answers as context for dependent Requirements
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..core import ResearchGoal, RequirementAnswer
from ..external_apis import LLMClient
from ..integrations.rosetta import RosettaTaskManager
from ..memory import ContextMemory
from ..mcp import MCPServerManager
from .configuration_agent import ConfigurationAgent
from .generation_agent import GenerationAgent
# v6.0: Simplified evaluation pipeline
from .reflection_coach_agent import ReflectionCoachAgent
from .tournament_ranking_agent import TournamentRankingAgent
from .evolution_architect_agent import EvolutionArchitectAgent


class SupervisorAgent:
    """
    Sequential Confirmation Research Orchestrator (v6.0)

    Architecture (RequirementAnswer-based):
    1. Configuration Phase: Parse research goal → Requirements with dependencies
    2. Execution Order: Topological sort of Requirements
    3. For each Requirement Group (parallel within, sequential between):
       - Generate N diverse RequirementAnswers
       - [PARALLEL] Reflection (4-criteria feedback) + Tournament (pairwise ranking)
       - Confirm best answer (winner from tournament)
       - Evolution (PASS - disabled in v6.0)
    4. Final: Assemble confirmed answers into research solution

    v6.0 Changes:
    - Removed Phase 1 (Pre-Check)
    - Reflection + Tournament run in parallel
    - ELO removed, pure win-count ranking
    - Evolution disabled (returns None)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Setup Supervisor-specific logging with session directory
        session_dir = config.get("session_dir")
        if session_dir:
            self.log_dir = Path(session_dir)
        else:
            self.log_dir = Path("./logs")
            self.log_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_supervisor_logger()
        
        self.logger.info("="*80)
        self.logger.info("🚀 SUPERVISOR AGENT INITIALIZING")
        self.logger.info("="*80)
        
        # Initialize clients
        self._init_clients()
        
        # Initialize memory
        self.memory = ContextMemory(config.get("storage_path", "./research_memory"))
        self.logger.info(f"✅ Context Memory initialized")

        # v6.0: Background enrichment disabled (Evolution PASS mode)
        # Retained for future use
        self.enrichment_tasks = []

        # Set RAs directory for config export
        session_dir = config.get("session_dir")
        if session_dir:
            import os
            ras_dir = os.path.join(session_dir, "RAs")
            self.memory.set_ras_directory(ras_dir)
            self.logger.info(f"✅ RAs directory set: {ras_dir}")

        # Initialize MCP Server Manager (will be initialized async in _initialize_research)
        mcp_config = config.get("mcp", {})
        enabled_servers = mcp_config.get("enabled_servers", None)  # None = all servers
        self.mcp_manager = MCPServerManager(enabled_servers=enabled_servers)
        self.logger.info(f"✅ MCP Server Manager created (will initialize in async context)")

        # Tool Registry and Prompt Manager (will be initialized after MCP servers start)
        from ..tools import ToolRegistry
        from ..prompts import PromptManager

        self.tool_registry = None  # Deferred - created after MCP init
        self.prompt_manager = PromptManager()
        self.logger.info(f"✅ Prompt Manager initialized")
        
        # Configuration (initialized in _initialize_research)
        self.configuration_agent = None
        self.research_config = None

        self.start_time = None

        self.logger.info("✅ Supervisor Agent initialized")
    
    def _setup_supervisor_logger(self) -> logging.Logger:
        """Setup dedicated logger for Supervisor - only essential flow and errors"""
        logger = logging.getLogger("SupervisorAgent")
        logger.setLevel(logging.INFO)  # Log INFO and above only

        # Remove existing handlers
        logger.handlers = []

        # File handler - Supervisor log file in session directory (essential flow only)
        log_file = self.log_dir / "supervisor.log"

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)  # INFO+: task creation, convergence checks, phase transitions

        # Formatter - concise format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        # Enable propagation to root logger so full_terminal.log also receives supervisor logs
        # This means supervisor logs go to BOTH supervisor.log (essential only) AND full_terminal.log (all logs)
        logger.propagate = True

        return logger
    
    def _init_clients(self):
        """Initialize external API clients"""
        # LLM Client
        llm_config = self.config.get("llm", {})
        try:
            self.llm_client = LLMClient(
                provider=llm_config.get("provider", "openrouter"),
                model=llm_config.get("model"),
                api_key=llm_config.get("api_key"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 8192)
            )
            self.logger.info(f"✅ LLM Client: {llm_config.get('model')}")
        except Exception as e:
            self.logger.warning(f"⚠️ LLM Client init failed: {e}")
            self.llm_client = None
        
        # Rosetta Task Manager
        rosetta_config = self.config.get("rosetta", {})
        try:
            self.rosetta_manager = RosettaTaskManager(
                max_concurrent=rosetta_config.get("max_concurrent", 2),
                python_path=rosetta_config.get("python_path", "python3"),
                timeout=rosetta_config.get("timeout", 3600)
            )
            self.logger.info(f"✅ Rosetta Task Manager initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Rosetta Task Manager init failed: {e}")
            self.rosetta_manager = None

    async def _initialize_research(
        self,
        research_goal: ResearchGoal,
        user_preferences: Optional[Dict[str, Any]]
    ):
        """
        Phase 0: Initialize research infrastructure for Sequential Confirmation.

        Steps:
        1. Initialize MCP Servers for data collection
        2. Create Tool Registry
        3. ConfigurationAgent: Parse goal → Research Plan Configuration
        4. Store research goal in memory
        """
        self.logger.info("-" * 80)
        self.logger.info("📋 PHASE 0: CONFIGURATION")
        self.logger.info("-" * 80)

        # Step 1: Initialize MCP Servers (async)
        self.logger.info("1️⃣ Initializing MCP Servers...")
        try:
            await self.mcp_manager.initialize()
            self.logger.info(f"✅ MCP Servers initialized: {len(self.mcp_manager.clients)} servers connected")
        except Exception as e:
            self.logger.error(f"⚠️ MCP Server initialization failed: {e}")
            self.logger.warning("Continuing without MCP tools - data collection will be limited")

        # Step 2: Create Tool Registry with MCP Manager
        from ..tools import ToolRegistry
        self.tool_registry = ToolRegistry(mcp_server_manager=self.mcp_manager)
        self.logger.info(f"✅ Tool Registry initialized ({len(self.tool_registry._tools)} tools available)")

        # Step 3: Configuration Agent - parse research goal (or use pre-parsed config)
        if self.research_config:
            # Pre-parsed configuration provided (e.g., from test_from_config.py)
            self.logger.info("✅ Using pre-parsed configuration (skipping ConfigurationAgent)")
        else:
            # Parse research goal using ConfigurationAgent
            self.configuration_agent = ConfigurationAgent(
                llm_client=self.llm_client
            )

            # Create configuration and save to log directory
            self.research_config = await self.configuration_agent.create_configuration(
                research_goal=research_goal,
                user_preferences=user_preferences or {},
                save_path=self.log_dir  # Save to log directory
            )

        # Store research goal in memory
        self.memory.set_research_goal(research_goal)

        # Log configuration details
        self.logger.info(f"✅ Research Configuration created:")

        # Get original research goal and parsed problem
        original_goal = self.research_config.get('original_research_goal', {})
        parsed_problem = self.research_config.get('parsed_problem', {})

        self.logger.info(f"   Project: {original_goal.get('title', 'N/A')}")
        self.logger.info(f"   Description: {original_goal.get('raw_text', 'N/A')[:100]}...")
        self.logger.info(f"   Type: {parsed_problem.get('problem_type', 'N/A')}")

        # Log requirements info
        requirements = parsed_problem.get('requirements', [])
        execution_order = parsed_problem.get('execution_order', [])
        self.logger.info(f"   Requirements: {len(requirements)}")
        self.logger.info(f"   Execution Order: {execution_order}")

        self.logger.info("-" * 80)
        self.logger.info("✅ INITIALIZATION COMPLETE")
        self.logger.info("-" * 80)

    async def _cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")

        # Close MCP servers
        if self.mcp_manager:
            try:
                await self.mcp_manager.close()
                self.logger.info("✅ MCP servers closed")
            except Exception as e:
                self.logger.warning(f"Error closing MCP servers: {e}")

        self.logger.info("✅ Cleanup complete")
        self.logger.info("=" * 80)

    async def run_sequential_confirmation(
        self,
        research_goal: ResearchGoal,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute research using Sequential Confirmation architecture.

        This is the NEW entry point for RequirementAnswer-based research.
        Instead of generating hypotheses, this:
        1. Parses requirements with dependencies
        2. Processes requirements in topological order
        3. Confirms each answer before moving to dependent requirements
        4. Optionally runs speculative exploration for diversity

        Args:
            research_goal: Research objective
            user_preferences: Optional user-specified preferences

        Returns:
            Final research results with confirmed answers
        """
        self.start_time = datetime.now()

        self.logger.info("=" * 80)
        self.logger.info("🔬 SEQUENTIAL CONFIRMATION RESEARCH STARTED")
        self.logger.info("=" * 80)

        try:
            # Phase 0: Configuration (reuse existing)
            await self._initialize_research(research_goal, user_preferences)

            # Phase 1: Sequential Confirmation Loop
            await self._execute_sequential_confirmation()

            # Phase 2: Finalization
            results = await self._finalize_sequential_research()

            return results

        except Exception as e:
            self.logger.error(f"❌ FATAL ERROR in sequential confirmation: {e}", exc_info=True)
            raise
        finally:
            await self._cleanup()

    async def _execute_sequential_confirmation(self):
        """
        Execute Sequential Confirmation: Process requirements in dependency order.

        Key principles:
        1. Topological sort for execution order
        2. Parallel processing within groups (no dependencies)
        3. Sequential processing between groups (dependencies)
        4. Confirm answers before proceeding to dependents
        """
        self.logger.info("-" * 80)
        self.logger.info("📋 PHASE 1: SEQUENTIAL CONFIRMATION")
        self.logger.info("-" * 80)

        # Get parsed problem from research config
        parsed_problem = self.research_config.get("parsed_problem", {})
        if not parsed_problem:
            self.logger.warning("No parsed_problem found in research_config")
            return

        # Get execution order (topological sort)
        execution_order = parsed_problem.get("execution_order", [])
        if not execution_order:
            # Fallback: compute topological sort from dependency graph
            self.logger.info(f"Failed to load computed execution order from parsed_problem")
            requirements = parsed_problem.get("requirements", [])
            execution_order = self._compute_execution_order(requirements)
            self.logger.info(f"Computed execution order from dependencies: {execution_order}")
        else:
            self.logger.info(f"Execution order from parsed_problem: {execution_order}")
        
        requirements_map = self._build_requirements_map(parsed_problem)
        self.logger.info(f"Total requirements: {len(requirements_map)}")

        # Create specialized agents for Sequential Confirmation
        generation_agent = GenerationAgent(
            memory=self.memory,
            config={
                **self.research_config,
                "parsed_problem": self.research_config.get("parsed_problem", {})
            },
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            mcp_server_manager=self.mcp_manager,
            experiment_dir=self.config.get("session_dir")  # NEW parameter
        )
        evolution_agent = EvolutionArchitectAgent(
            memory=self.memory,
            config={
                **self.research_config,
                "parsed_problem": self.research_config.get("parsed_problem", {})
            },
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            mcp_server_manager=self.mcp_manager
        )
        # Note: Absolute/Relative evaluation agents are created inside _process_single_requirement()

        # Process each group
        for group_idx, requirement_group in enumerate(execution_order):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"📦 GROUP {group_idx + 1}/{len(execution_order)}: {requirement_group}")
            self.logger.info(f"{'=' * 60}")

            # Process requirements in this group in parallel
            tasks = []
            for req_id in requirement_group:
                if req_id not in requirements_map:
                    self.logger.warning(f"Requirement {req_id} not found in map, skipping")
                    continue

                requirement = requirements_map[req_id]

                # Get context from confirmed dependencies
                try:
                    depends_on = requirement.get("depends_on", [])
                    context = self.memory.get_context_for_requirement(req_id, depends_on)
                except ValueError as e:
                    self.logger.error(f"Missing dependencies for {req_id}: {e}")
                    continue

                # Create task for this requirement
                tasks.append(
                    self._process_single_requirement(
                        requirement=requirement,
                        context=context,
                        generation_agent=generation_agent,
                        evolution_agent=evolution_agent
                    )
                )

            # Execute all requirements in this group in parallel
            if tasks:
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    self.logger.error(f"✗ Group {group_idx + 1} failed: {e}")
                    raise RuntimeError(f"Sequential Confirmation failed at group {group_idx + 1}: {e}")

            self.logger.info(f"✅ Group {group_idx + 1} complete")

        # All requirements confirmed
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ ALL REQUIREMENTS CONFIRMED")
        self.logger.info(f"Total confirmed answers: {len(self.memory.confirmed_answers)}")
        self.logger.info("=" * 80)

        # v6.0: Evolution disabled, no enrichment tasks to wait for
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ SEQUENTIAL CONFIRMATION COMPLETE")
        self.logger.info("=" * 80)
#
    async def _process_single_requirement(
        self,
        requirement: Dict[str, Any],
        context: Dict[str, RequirementAnswer],
        generation_agent: GenerationAgent,
        evolution_agent: EvolutionArchitectAgent
    ):
        """
        Process a single requirement (v6.0):
        Generate → [Reflection + Tournament PARALLEL] → Confirm → Evolution (PASS)

        This is the core loop for each requirement in Sequential Confirmation.
        Reflection and Tournament run in parallel for efficiency.
        """
        req_id = requirement.get("requirement_id", requirement.get("step_id", "unknown"))
        req_title = requirement.get("title", "")

        self.logger.info(f"\n  ▶ Processing Requirement {req_id}: {req_title[:40]}...")

        # v6.0 Simplified Workflow: Generate → [Reflect + Rank] → Confirm

        # Step 1: Generation - Create N diverse answers
        self.logger.info(f"    1️⃣ Generating diverse answers...")
        gen_result = await generation_agent.run_for_requirement({
            "requirement": requirement,
            "num_answers": 3,
            "context": context,
            "research_goal": self.memory.get_research_goal()
        })

        if gen_result.get("status") != "success":
            self.logger.error(f"    ✗ Generation failed: {gen_result.get('message')}")
            raise RuntimeError(f"Failed to generate answers for requirement {req_id}")

        self.logger.info(f"    ✓ Generated {len(gen_result.get('answers', []))} answers")

        # Get generated answers from memory
        answers = self.memory.get_answers_for_requirement(req_id)
        generated_answers = [a for a in answers if a.status in ["generated", "reviewed"]]

        if not generated_answers:
            self.logger.error(f"    ✗ No answers to evaluate")
            raise RuntimeError(f"Failed to generate any answers for requirement {req_id}")

        # Create agents for v6.0 pipeline
        reflection_agent = ReflectionCoachAgent(
            memory=self.memory,
            config=self.research_config,
            llm_client=self.llm_client
        )
        tournament_ranker = TournamentRankingAgent(
            memory=self.memory,
            config=self.research_config,
            llm_client=self.llm_client
        )

        # v6.0: PARALLEL Reflection + Tournament
        self.logger.info(f"    2️⃣ Running Reflection + Tournament (PARALLEL)...")

        # Create parallel tasks
        reflection_task = asyncio.create_task(
            self._run_reflection_for_all(reflection_agent, generated_answers, requirement)
        )
        tournament_task = asyncio.create_task(
            tournament_ranker.run_tournament(
                answers=generated_answers,
                requirement=requirement
            )
        )

        # Wait for both to complete
        reflection_results, tournament_result = await asyncio.gather(
            reflection_task, tournament_task
        )

        # Attach reflection results to answers
        for answer in generated_answers:
            if answer.id in reflection_results:
                answer.metadata["reflection"] = reflection_results[answer.id].to_dict()
                # Log feedback summary
                weak_criteria = reflection_results[answer.id].get_weak_criteria()
                self.logger.info(
                    f"    │  ✓ {answer.id[:8]}...: "
                    f"{len(reflection_results[answer.id].feedback_items)} feedback items, "
                    f"{len(weak_criteria)} weak/missing"
                )
                self.memory.store_requirement_answer(answer)

        # Get ranked answers from tournament
        ranked_answers = tournament_result["ranked_answers"]
        self.logger.info(
            f"    ✓ Tournament complete. Rankings: "
            f"""{[(a.id[:8] + '...', a.tournament_rank, f'wins={getattr(a, "wins", 0)}') for a in ranked_answers[:3]]}"""
        )

        # Get best answer from tournament
        best_answer = ranked_answers[0] if ranked_answers else None

        if not best_answer:
            self.logger.error(f"    ✗ No answers available from tournament")
            raise RuntimeError(f"No answers available for requirement {req_id}")

        # Show confirmation status
        expected = requirement.get("expected_deliverables", [])
        actual = best_answer.deliverables if isinstance(best_answer.deliverables, dict) else {}
        deliverables_ratio = len(actual) / max(len(expected), 1)

        # Check if won via novelty tiebreaker
        is_novelty_winner = best_answer.metadata.get("is_novelty_winner", False)
        novelty_note = " (won by novelty)" if is_novelty_winner else ""

        self.logger.info(
            f"    ✅ Confirming best answer{novelty_note}\n"
            f"       Winner: {best_answer.id[:16]}...\n"
            f"       Wins: {getattr(best_answer, 'wins', 'N/A')}, "
            f"Losses: {getattr(best_answer, 'losses', 'N/A')}\n"
            f"       Deliverables: {deliverables_ratio:.1%}"
        )

        # Confirm the best answer
        self.memory.confirm_answer(best_answer.id)

        # v6.0: Evolution Phase (PASS - disabled)
        self.logger.info(f"    3️⃣ Evolution Phase: PASS (v6.0 - disabled)")
        # No enrichment task - evolution_agent.enrich_confirmed_answer() returns None immediately

    async def _run_reflection_for_all(
        self,
        reflection_agent: ReflectionCoachAgent,
        answers: List[RequirementAnswer],
        requirement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run reflection on all answers in parallel.

        Args:
            reflection_agent: ReflectionCoachAgent instance
            answers: List of answers to reflect on
            requirement: Requirement specification

        Returns:
            Dict mapping answer_id -> ReflectionResult
        """
        tasks = [
            reflection_agent.reflect_on_answer(answer, requirement)
            for answer in answers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reflection_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"    │  ✗ Reflection failed for {answers[i].id[:8]}...: {result}")
            else:
                reflection_results[answers[i].id] = result

        return reflection_results

    async def _run_enrichment_background(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any],
        context: Dict[str, RequirementAnswer],
        evolution_agent,
        req_id: str
    ):
        """
        Run enrichment in background without blocking next group.

        This method is spawned as an asyncio task and runs independently.
        """
        import time
        start_time = time.time()

        try:
            self.logger.info(f"    [{req_id}] 🔬 Enrichment starting...")

            # v5.0: Call enrich_confirmed_answer with three separate arguments
            enrichment_result = await evolution_agent.enrich_confirmed_answer(
                answer=answer,
                requirement=requirement,
                context=context
            )

            duration = time.time() - start_time

            if enrichment_result:
                # EnrichmentResult has protocol, literature, risk modules
                modules_completed = sum([
                    1 if enrichment_result.protocol else 0,
                    1 if enrichment_result.literature else 0,
                    1 if enrichment_result.risk else 0
                ])
                self.logger.info(
                    f"    [{req_id}] ✓ Enrichment complete ({duration:.1f}s): "
                    f"{modules_completed}/3 modules successful, "
                    f"confidence={enrichment_result.overall_confidence:.2f}"
                )

                # Store enrichment in answer metadata
                answer.metadata["enrichment"] = enrichment_result.to_dict()
                self.memory.store_requirement_answer(answer)
            else:
                self.logger.warning(
                    f"    [{req_id}] ⚠️ Enrichment returned None ({duration:.1f}s)"
                )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.warning(f"    [{req_id}] ⚠️ Enrichment error ({duration:.1f}s): {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            # Enrichment failure does not break confirmation (graceful degradation)

    def _check_answer_convergence(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any]
    ) -> bool:
        """
        Check if an answer meets convergence criteria (v6.0: Simplified).

        v6.0 Changes:
        - No quality_score (Reflection provides feedback, not scores)
        - No ELO rating (Tournament uses win-count)
        - Convergence is based only on deliverables coverage

        In v6.0, the best answer from tournament is always confirmed.
        This method is kept for backward compatibility but simplified.
        """
        # Read convergence criteria from execution_plan
        conv_config = self.research_config.get("execution_plan", {}).get("convergence", {})
        deliverables_threshold = conv_config.get("deliverables_ratio", 0.8)

        # Check deliverables coverage
        expected = requirement.get("expected_deliverables", [])
        actual = answer.deliverables if isinstance(answer.deliverables, dict) else {}
        deliverables_ratio = len(actual) / max(len(expected), 1)
        deliverables_ok = deliverables_ratio >= deliverables_threshold

        return deliverables_ok
#
    def _build_requirements_map(self, parsed_problem: Dict[str, Any]) -> Dict[str, Dict]:
        """Build a map of requirement_id -> requirement dict"""
        requirements = parsed_problem.get("requirements", [])
        return {
            r.get("requirement_id", r.get("step_id", f"req_{i}")): r
            for i, r in enumerate(requirements)
        }
#
    def _compute_execution_order(self, requirements: List[Dict]) -> List[List[str]]:
        """
        Compute execution order using topological sort based on dependencies.

        Groups requirements by level - requirements in the same group have no
        dependencies on each other and can be processed in parallel.

        Example:
            Input: [
                {"requirement_id": "1", "depends_on": []},
                {"requirement_id": "2", "depends_on": ["1"]},
                {"requirement_id": "3", "depends_on": ["2"]},
                {"requirement_id": "4", "depends_on": ["3"]},
                {"requirement_id": "5", "depends_on": ["3"]},
                {"requirement_id": "6", "depends_on": ["4", "5"]}
            ]
            Output: [["1"], ["2"], ["3"], ["4", "5"], ["6"]]
                    - 4 and 5 can run in parallel (both depend only on 3)

        Returns:
            List of groups, where each group contains requirement IDs that can run in parallel
        """
        from collections import defaultdict, deque

        # Build adjacency list and in-degree map
        in_degree = {}
        dependents = defaultdict(list)  # req_id -> list of reqs that depend on it
        all_ids = set()

        for req in requirements:
            req_id = req.get("requirement_id", req.get("step_id", ""))
            deps = req.get("depends_on", [])
            all_ids.add(req_id)
            in_degree[req_id] = len(deps)
            for dep in deps:
                dependents[dep].append(req_id)

        # Kahn's algorithm with level tracking
        execution_order = []
        current_level = [req_id for req_id in all_ids if in_degree[req_id] == 0]

        while current_level:
            # Sort for deterministic order
            current_level.sort()
            execution_order.append(current_level)

            next_level = []
            for req_id in current_level:
                for dependent in dependents[req_id]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_level.append(dependent)

            current_level = next_level

        # Check for cycles (unprocessed nodes)
        processed = set()
        for group in execution_order:
            processed.update(group)

        if len(processed) < len(all_ids):
            unprocessed = all_ids - processed
            self.logger.warning(f"Circular dependency detected! Unprocessed: {unprocessed}")
            # Fallback: add remaining as final group
            execution_order.append(list(unprocessed))

        return execution_order

    async def _finalize_sequential_research(self) -> Dict[str, Any]:
        """Finalize Sequential Confirmation research and generate results (v6.0)."""
        self.logger.info("-" * 80)
        self.logger.info("📊 PHASE 2: FINALIZATION")
        self.logger.info("-" * 80)

        # Collect all confirmed answers
        confirmed = self.memory.get_all_confirmed_answers()

        # Build final solution (v6.0: no ELO, use wins/losses)
        final_solution = {}
        for req_id, answer in confirmed.items():
            final_solution[req_id] = {
                "requirement_id": req_id,
                "requirement_title": getattr(answer, "requirement_title", ""),
                "answer": answer.answer,
                "rationale": answer.rationale,
                "deliverables": answer.deliverables,
                "confidence": getattr(answer, "confidence", 0.0),
                "wins": getattr(answer, "wins", 0),
                "losses": getattr(answer, "losses", 0),
                "tournament_rank": getattr(answer, "tournament_rank", 1),
                "is_novelty_winner": answer.metadata.get("is_novelty_winner", False),
                "reflection": answer.metadata.get("reflection", {})
            }

        # Calculate overall stats (v6.0: no quality/ELO averages)
        total_answers = len(self.memory.requirement_answers)
        confirmed_count = len(confirmed)

        duration = (datetime.now() - self.start_time).total_seconds()

        self.logger.info(f"✅ Research Complete:")
        self.logger.info(f"   - Requirements answered: {confirmed_count}")
        self.logger.info(f"   - Total answers generated: {total_answers}")
        self.logger.info(f"   - Duration: {duration:.1f}s")

        return {
            "status": "success",
            "mode": "sequential_confirmation",
            "confirmed_answers": final_solution,
            "statistics": {
                "total_requirements": len(final_solution),
                "total_answers_generated": total_answers,
                "duration_seconds": duration
            },
            "metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "version": "v6.0"
            }
        }
