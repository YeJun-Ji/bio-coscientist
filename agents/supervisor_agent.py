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
from .reflection_agent import ReflectionAgent
from .ranking_agent import RankingAgent
from .evolution_agent import EvolutionAgent


class SupervisorAgent:
    """
    Sequential Confirmation Research Orchestrator

    Architecture (RequirementAnswer-based):
    1. Configuration Phase: Parse research goal → Requirements with dependencies
    2. Execution Order: Topological sort of Requirements
    3. For each Requirement Group (parallel within, sequential between):
       - Generate N diverse RequirementAnswers
       - Reflect (evaluate) each answer
       - Rank answers via ELO tournament
       - Evolve if not converged
       - Confirm best answer
    4. Final: Assemble confirmed answers into research solution
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
        logger.setLevel(logging.INFO)  # INFO 이상만 기록

        # Remove existing handlers
        logger.handlers = []

        # File handler - Supervisor log file in session directory (essential flow only)
        log_file = self.log_dir / "supervisor.log"

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)  # INFO 이상: 태스크 생성, 수렴 체크, 단계 전환

        # Formatter - 간결한 형식
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

    # ========================================================================
    # SEQUENTIAL CONFIRMATION: RequirementAnswer-based Research Orchestration
    # ========================================================================

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

        # Agent 생성
        # Create specialized agents for Sequential Confirmation
        generation_agent = self._create_sc_generation_agent()
        reflection_agent = self._create_sc_reflection_agent()
        ranking_agent = self._create_sc_ranking_agent()
        evolution_agent = self._create_sc_evolution_agent()

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
                        reflection_agent=reflection_agent,
                        ranking_agent=ranking_agent,
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

        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ SEQUENTIAL CONFIRMATION COMPLETE")
        self.logger.info(f"Total confirmed answers: {len(self.memory.confirmed_answers)}")
        self.logger.info("=" * 80)

    async def _process_single_requirement(
        self,
        requirement: Dict[str, Any],
        context: Dict[str, RequirementAnswer],
        generation_agent: GenerationAgent,
        reflection_agent: ReflectionAgent,
        ranking_agent: RankingAgent,
        evolution_agent: EvolutionAgent
    ):
        """
        Process a single requirement: Generate → Reflect → Rank → (Evolve) → Confirm

        This is the core loop for each requirement in Sequential Confirmation.
        """
        req_id = requirement.get("requirement_id", requirement.get("step_id", "unknown"))
        req_title = requirement.get("title", "")

        self.logger.info(f"\n  ▶ Processing Requirement {req_id}: {req_title[:40]}...")

        # Read max_iterations from execution_plan (v3.0 config)
        max_iterations = self.research_config.get("execution_plan", {}).get("per_requirement", {}).get("max_iterations", 5)

        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"    Iteration {iteration}/{max_iterations}")

            # Step 1: Generation - Create N diverse answers
            if iteration == 1:
                self.logger.info(f"    1️⃣ Generating diverse answers...")
                gen_result = await generation_agent.run_for_requirement({
                    "requirement": requirement,
                    "num_answers": 3,
                    "context": context,
                    "research_goal": self.memory.get_research_goal()
                })

                if gen_result.get("status") != "success":
                    self.logger.error(f"    ✗ Generation failed: {gen_result.get('message')}")
                    break

                self.logger.info(f"    ✓ Generated {len(gen_result.get('answers', []))} answers")

            # Step 2: Reflection - Evaluate all unreviewed answers (parallel)
            answers = self.memory.get_answers_for_requirement(req_id)
            unreviewed = [a for a in answers if a.status == "generated"]

            if unreviewed:
                self.logger.info(f"    2️⃣ Reviewing {len(unreviewed)} answers...")
                reflection_tasks = [
                    reflection_agent.run_for_answer({
                        "answer": answer,
                        "requirement": requirement,
                        "context": context
                    })
                    for answer in unreviewed
                ]
                await asyncio.gather(*reflection_tasks)
                self.logger.info(f"    ✓ Reviews complete")

            # Step 3: Ranking - Tournament among reviewed answers
            reviewed = self.memory.get_reviewed_answers(req_id)
            if len(reviewed) >= 2:
                self.logger.info(f"    3️⃣ Ranking {len(reviewed)} answers...")
                await ranking_agent.run_for_answers({
                    "requirement_id": req_id,
                    "answers": reviewed,
                    "requirement": requirement
                })
                self.logger.info(f"    ✓ Ranking complete")

            # Step 4: Check convergence
            all_answers = self.memory.get_answers_for_requirement(req_id)
            best_answer = max(all_answers, key=lambda a: a.elo_rating) if all_answers else None

            if best_answer and self._check_answer_convergence(best_answer, requirement):
                # Confirm this answer!
                self.memory.confirm_answer(best_answer.id)
                self.logger.info(f"    ✅ CONFIRMED: {best_answer.id} (ELO: {best_answer.elo_rating:.1f})")
                return

            # Step 5: Not converged → Evolution
            if best_answer and iteration < max_iterations:
                self.logger.info(f"    5️⃣ Evolving best answer...")
                evo_result = await evolution_agent.run_for_answer({
                    "answer": best_answer,
                    "requirement": requirement,
                    "context": context,
                    "method": "grounding"
                })

                if evo_result.get("status") == "success":
                    self.logger.info(f"    ✓ Created evolved answer")
                # Continue to next iteration to evaluate evolved answer

        # Max iterations reached - force confirm best answer
        all_answers = self.memory.get_answers_for_requirement(req_id)
        if all_answers:
            best_answer = max(all_answers, key=lambda a: a.elo_rating)
            self.memory.confirm_answer(best_answer.id)
            self.logger.info(f"    ⚠️ FORCE CONFIRMED (max iterations): {best_answer.id}")
        else:
            self.logger.error(f"    ✗ No answers generated for {req_id}")
            raise RuntimeError(f"Failed to generate any answers for requirement {req_id}")

    def _check_answer_convergence(
        self,
        answer: RequirementAnswer,
        requirement: Dict[str, Any]
    ) -> bool:
        """
        Check if an answer meets convergence criteria.

        Criteria (read from execution_plan.convergence):
        1. Quality score >= min_quality_score
        2. ELO rating >= min_elo_rating
        3. Delivers >= deliverables_ratio of expected deliverables
        """
        # Read convergence criteria from execution_plan (v3.0 config)
        conv_config = self.research_config.get("execution_plan", {}).get("convergence", {})
        quality_threshold = conv_config.get("min_quality_score", 0.7)
        elo_threshold = conv_config.get("min_elo_rating", 1300)
        deliverables_threshold = conv_config.get("deliverables_ratio", 0.8)

        quality_ok = answer.quality_score >= quality_threshold
        elo_ok = answer.elo_rating >= elo_threshold

        # Check deliverables coverage
        expected = requirement.get("expected_deliverables", [])
        actual = answer.deliverables if isinstance(answer.deliverables, dict) else {}
        deliverables_ratio = len(actual) / max(len(expected), 1)
        deliverables_ok = deliverables_ratio >= deliverables_threshold

        converged = quality_ok and elo_ok and deliverables_ok

        if not converged:
            self.logger.debug(
                f"    Convergence check: quality={answer.quality_score:.2f}>={quality_threshold}, "
                f"elo={answer.elo_rating:.1f}>={elo_threshold}, "
                f"deliverables={deliverables_ratio:.1%}>={deliverables_threshold:.0%}"
            )

        return converged

    def _build_requirements_map(self, parsed_problem: Dict[str, Any]) -> Dict[str, Dict]:
        """Build a map of requirement_id -> requirement dict"""
        requirements = parsed_problem.get("requirements", [])
        return {
            r.get("requirement_id", r.get("step_id", f"req_{i}")): r
            for i, r in enumerate(requirements)
        }

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

    def _create_sc_generation_agent(self) -> GenerationAgent:
        """Create GenerationAgent for Sequential Confirmation"""
        return GenerationAgent(
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

    def _create_sc_reflection_agent(self) -> ReflectionAgent:
        """Create ReflectionAgent for Sequential Confirmation"""
        return ReflectionAgent(
            memory=self.memory,
            config={
                **self.research_config,
                "parsed_problem": self.research_config.get("parsed_problem", {})
            },
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            mcp_server_manager=self.mcp_manager
        )

    def _create_sc_ranking_agent(self) -> RankingAgent:
        """Create RankingAgent for Sequential Confirmation"""
        return RankingAgent(
            memory=self.memory,
            config={
                **self.research_config,
                "parsed_problem": self.research_config.get("parsed_problem", {})
            },
            llm_client=self.llm_client
        )

    def _create_sc_evolution_agent(self) -> EvolutionAgent:
        """Create EvolutionAgent for Sequential Confirmation"""
        return EvolutionAgent(
            memory=self.memory,
            config={
                **self.research_config,
                "parsed_problem": self.research_config.get("parsed_problem", {})
            },
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            mcp_server_manager=self.mcp_manager
        )

    async def _finalize_sequential_research(self) -> Dict[str, Any]:
        """Finalize Sequential Confirmation research and generate results."""
        self.logger.info("-" * 80)
        self.logger.info("📊 PHASE 2: FINALIZATION")
        self.logger.info("-" * 80)

        # Collect all confirmed answers
        confirmed = self.memory.get_all_confirmed_answers()

        # Build final solution
        final_solution = {}
        for req_id, answer in confirmed.items():
            final_solution[req_id] = {
                "requirement_id": req_id,
                "requirement_title": answer.requirement_title,
                "answer": answer.answer,
                "rationale": answer.rationale,
                "deliverables": answer.deliverables,
                "confidence": answer.confidence,
                "elo_rating": answer.elo_rating,
                "quality_score": answer.quality_score,
                "iteration": answer.iteration
            }

        # Calculate overall stats
        total_answers = len(self.memory.requirement_answers)
        confirmed_count = len(confirmed)
        avg_quality = sum(a.quality_score for a in confirmed.values()) / max(confirmed_count, 1)
        avg_elo = sum(a.elo_rating for a in confirmed.values()) / max(confirmed_count, 1)

        duration = (datetime.now() - self.start_time).total_seconds()

        self.logger.info(f"✅ Research Complete:")
        self.logger.info(f"   - Requirements answered: {confirmed_count}")
        self.logger.info(f"   - Total answers generated: {total_answers}")
        self.logger.info(f"   - Average quality: {avg_quality:.2f}")
        self.logger.info(f"   - Average ELO: {avg_elo:.1f}")
        self.logger.info(f"   - Duration: {duration:.1f}s")

        return {
            "status": "success",
            "mode": "sequential_confirmation",
            "confirmed_answers": final_solution,
            "statistics": {
                "total_requirements": len(final_solution),
                "total_answers_generated": total_answers,
                "average_quality": avg_quality,
                "average_elo": avg_elo,
                "duration_seconds": duration
            },
            "metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        }
