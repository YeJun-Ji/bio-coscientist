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
from .log_verification_agent import LogVerificationAgent
# v5.0: New agents replace old evaluation pipeline
from .reflection_coach_agent import ReflectionCoachAgent
from .tournament_ranking_agent import TournamentRankingAgent
from .evolution_architect_agent import EvolutionArchitectAgent
# Deprecated (v4.0): Keep imports for backward compatibility warnings
from .quality_assessment_agent import QualityAssessmentAgent  # DEPRECATED v5.0
from .ranking_agent import RankingAgent  # DEPRECATED v5.0
from .evolution_agent import EvolutionAgent  # DEPRECATED v5.0, use EvolutionArchitectAgent


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

        # Track background enrichment tasks (v3.0)
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

        # All requirements confirmed - now wait for background enrichments
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ ALL REQUIREMENTS CONFIRMED")
        self.logger.info(f"Total confirmed answers: {len(self.memory.confirmed_answers)}")
        self.logger.info("=" * 80)

        if self.enrichment_tasks:
            self.logger.info(f"\n🔬 Waiting for {len(self.enrichment_tasks)} background enrichments to complete...")
            enrichment_start = asyncio.get_event_loop().time()

            # Gather all enrichment results (already running in background)
            await asyncio.gather(*self.enrichment_tasks, return_exceptions=True)

            enrichment_duration = asyncio.get_event_loop().time() - enrichment_start
            self.logger.info(f"✓ All enrichments complete ({enrichment_duration:.1f}s)")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ SEQUENTIAL CONFIRMATION COMPLETE (with enrichments)")
        self.logger.info("=" * 80)
#
    async def _process_single_requirement(
        self,
        requirement: Dict[str, Any],
        context: Dict[str, RequirementAnswer],
        generation_agent: GenerationAgent,
        evolution_agent: EvolutionArchitectAgent  # v5.0: Changed from EvolutionAgent
    ):
        """
        Process a single requirement: Generate → [Absolute + Relative Evaluation] → (Evolve) → Confirm

        This is the core loop for each requirement in Sequential Confirmation.
        Evaluation agents are created inside this method for parallel execution.
        """
        req_id = requirement.get("requirement_id", requirement.get("step_id", "unknown"))
        req_title = requirement.get("title", "")

        self.logger.info(f"\n  ▶ Processing Requirement {req_id}: {req_title[:40]}...")

        # v3.0 Simplified Workflow: Generate → Evaluate → Converge/Confirm
        # No multi-iteration loops - quality_score from single evaluation determines outcome

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

        # Step 2-5: v5.0 Simplified Evaluation Pipeline
        # PHASE 1: Pre-Check (Fast-Fail - KEEP)
        # PHASE 2: Reflection (NEW - Review + Feedback, NO regeneration)
        # PHASE 3: Tournament (NEW - Relative evaluation)
        # PHASE 4: Evolution (NEW - Enrich confirmed winner only)
        answers = self.memory.get_answers_for_requirement(req_id)
        generated_answers = [a for a in answers if a.status in ["generated", "reviewed"]]

        if not generated_answers:
            self.logger.error(f"    ✗ No answers to evaluate")
            raise RuntimeError(f"Failed to generate any answers for requirement {req_id}")

        # Create agents for v5.0 pipeline
        from .log_verification_agent import LogVerificationAgent
        from .reflection_coach_agent import ReflectionCoachAgent
        from .tournament_ranking_agent import TournamentRankingAgent

        log_verifier = LogVerificationAgent(
            memory=self.memory,
            config=self.research_config,
            llm_client=self.llm_client,
            experiment_dir=self.config.get("session_dir")
        )
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

        # PHASE 1: Pre-Check (Fast-Fail, <0.5s each - UNCHANGED)
        self.logger.info(f"    2️⃣ PHASE 1: Pre-Check (fast-fail)...")
        valid_answers = []
        for answer in generated_answers:
            try:
                is_valid = log_verifier.pre_check(answer, req_id, self.config.get("session_dir"))
                if is_valid:
                    valid_answers.append(answer)
                else:
                    self.logger.warning(f"    │  ✗ {answer.id[:8]}... rejected in pre-check")
            except Exception as e:
                self.logger.warning(f"    │  ✗ {answer.id[:8]}... pre-check failed: {e}")

        if not valid_answers:
            self.logger.error(f"    ✗ All answers failed pre-check")
            raise RuntimeError(f"All answers failed pre-check for requirement {req_id}")

        self.logger.info(f"    ✓ {len(valid_answers)}/{len(generated_answers)} answers passed pre-check")

        # PHASE 2: Reflection (NEW - Review all answers, NO regeneration)
        self.logger.info(f"    3️⃣ PHASE 2: Reflection (coach-style review)...")

        # Reflect on each answer (can be parallel)
        reflection_tasks = []
        for answer in valid_answers:
            # Get verification results first
            verification = log_verifier.verify(answer, req_id, self.config.get("session_dir"))
            # Create reflection task
            task = reflection_agent.reflect_on_answer(
                answer=answer,
                requirement=requirement,
                verification_results=verification if not isinstance(verification, Exception) else {},
                config=self.research_config
            )
            reflection_tasks.append((answer, task))

        # Wait for all reflections
        for answer, task in reflection_tasks:
            try:
                reflection_result = await task
                answer.metadata["reflection"] = reflection_result.to_dict()
                answer.quality_score = reflection_result.overall_score
                answer.verification_score = reflection_result.verification_score
                self.memory.store_requirement_answer(answer)  # Update with reflection metadata
                self.logger.info(
                    f"    │  ✓ {answer.id[:8]}...: score={reflection_result.overall_score:.3f} "
                    f"(feedback items: {len(reflection_result.actionable_feedback)})"
                )
            except Exception as e:
                self.logger.error(f"    │  ✗ Reflection failed for {answer.id[:8]}...: {e}")
                answer.quality_score = 0.5  # Fallback

        scores = [a.quality_score for a in valid_answers]
        self.logger.info(f"    ✓ Reflection complete: scores = {[f'{s:.2f}' for s in scores]}")

        # Check if all scores are very low
        if all(s < 0.5 for s in scores):
            self.logger.warning(
                f"    ⚠️ All answers have low quality scores (< 0.5). "
                f"Proceeding with tournament, but results may be unreliable."
            )

        # PHASE 3: Tournament (NEW - Rank ALL valid answers)
        self.logger.info(f"    4️⃣ PHASE 3: Tournament Ranking...")
        tournament_result = await tournament_ranker.run_tournament(
            answers=valid_answers,
            requirement=requirement
        )

        ranked_answers = tournament_result["ranked_answers"]
        self.logger.info(
            f"    ✓ Tournament complete. Rankings: "
            f"{[(a.id[:8] + '...', a.tournament_rank, f'{a.elo_rating:.0f}') for a in ranked_answers[:3]]}"
        )

        # Step 4: Get best answer from tournament
        best_answer = ranked_answers[0] if ranked_answers else None

        if not best_answer:
            self.logger.error(f"    ✗ No answers available from tournament")
            raise RuntimeError(f"No answers available for requirement {req_id}")

        # Show detailed convergence status
        expected = requirement.get("expected_deliverables", [])
        actual = best_answer.deliverables if isinstance(best_answer.deliverables, dict) else {}
        deliverables_ratio = len(actual) / max(len(expected), 1)

        converged = self._check_answer_convergence(best_answer, requirement)

        if converged:
            self.logger.info(
                f"    ✅ CONVERGED - Confirming answer\n"
                f"       Quality: {best_answer.quality_score:.2f} (threshold: 0.7) ✓\n"
                f"       Deliverables: {deliverables_ratio:.1%} (threshold: 80%) ✓\n"
                f"       Best answer: {best_answer.id} (ELO: {best_answer.elo_rating:.1f})"
            )
        else:
            self.logger.info(
                f"    ⚠️ NOT CONVERGED - Confirming best available answer\n"
                f"       Quality: {best_answer.quality_score:.2f} (threshold: 0.7) {'✓' if best_answer.quality_score >= 0.7 else '✗'}\n"
                f"       Deliverables: {deliverables_ratio:.1%} (threshold: 80%) {'✓' if deliverables_ratio >= 0.8 else '✗'}\n"
                f"       Best answer: {best_answer.id} (ELO: {best_answer.elo_rating:.1f})"
            )

        # Confirm the best answer (regardless of convergence - v3.0 always confirms)
        self.memory.confirm_answer(best_answer.id)

        # Post-confirmation enrichment (v3.0) - RUN IN BACKGROUND
        # Don't await! Let next group start immediately.
        self.logger.info(f"    🔬 Starting post-confirmation enrichment (background)...")
        enrichment_task = asyncio.create_task(
            self._run_enrichment_background(
                answer=best_answer,
                requirement=requirement,
                context=context,
                evolution_agent=evolution_agent,
                req_id=req_id
            )
        )
        self.enrichment_tasks.append(enrichment_task)
        self.logger.info(f"    ↻ Enrichment running in background (not blocking next group)")

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
        Check if an answer meets convergence criteria (v3.0: Quality-Score Based).

        NEW (v3.0): Primary convergence is based on QUALITY_SCORE from AbsoluteEvaluationAgent.
        ELO rating is used for RANKING only, not convergence.

        Criteria (read from execution_plan.convergence):
        1. Quality score >= min_quality_score (PRIMARY - from 4-lens absolute evaluation)
        2. Delivers >= deliverables_ratio of expected deliverables
        3. ELO rating >= min_elo_rating (OPTIONAL - disabled by default in v3.0)

        The v3.0 philosophy: Quality is absolute (via tool/completeness/interpretability/consistency lenses),
        ranking is relative (via pairwise ELO). Use quality for convergence, ELO for selection.
        """
        # Read convergence criteria from execution_plan (v3.0 config)
        conv_config = self.research_config.get("execution_plan", {}).get("convergence", {})
        quality_threshold = conv_config.get("min_quality_score", 0.7)
        deliverables_threshold = conv_config.get("deliverables_ratio", 0.8)

        # ELO threshold now OPTIONAL (set to 0 to disable ELO-based convergence)
        elo_threshold = conv_config.get("min_elo_rating", 0)  # Default: 0 (disabled)
        use_elo_for_convergence = elo_threshold > 0

        # Check quality score (PRIMARY)
        quality_ok = answer.quality_score >= quality_threshold

        # Check deliverables coverage
        expected = requirement.get("expected_deliverables", [])
        actual = answer.deliverables if isinstance(answer.deliverables, dict) else {}
        deliverables_ratio = len(actual) / max(len(expected), 1)
        deliverables_ok = deliverables_ratio >= deliverables_threshold

        # Check ELO (OPTIONAL - only if explicitly enabled)
        elo_ok = True  # Default: pass
        if use_elo_for_convergence:
            elo_ok = answer.elo_rating >= elo_threshold

        # Converged if quality AND deliverables are met (ELO optional)
        converged = quality_ok and deliverables_ok and elo_ok

        if not converged:
            debug_msg = (
                f"    Convergence check: quality={answer.quality_score:.2f}>={quality_threshold} ({quality_ok}), "
                f"deliverables={deliverables_ratio:.1%}>={deliverables_threshold:.0%} ({deliverables_ok})"
            )
            if use_elo_for_convergence:
                debug_msg += f", elo={answer.elo_rating:.1f}>={elo_threshold} ({elo_ok})"
            else:
                debug_msg += f", elo={answer.elo_rating:.1f} (not used for convergence)"
            self.logger.debug(debug_msg)

        return converged
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
