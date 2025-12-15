"""
Supervisor Agent
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import Hypothesis, Review, ResearchGoal, HypothesisStatus, TournamentMatch
from ..clients import LLMClient, WebSearchClient, EmbeddingClient
from ..clients.rosetta_task_manager import RosettaTaskManager
from ..memory import ContextMemory
from .base_agent import BaseAgent
from .generation_agent import GenerationAgent
from .reflection_agent import ReflectionAgent
from .ranking_agent import RankingAgent
from .proximity_agent import ProximityAgent
from .evolution_agent import EvolutionAgent
from .metareview_agent import MetaReviewAgent

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Orchestrates the entire AI co-scientist workflow.
    
    Responsibilities:
    - Coordinate all specialized agents
    - Manage iterative research cycles
    - Assign tasks to workers
    - Monitor progress and performance
    - Generate final research overview
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = ContextMemory(config.get("storage_path", "./research_memory"))
        
        # Initialize LLM client
        llm_config = config.get("llm", {})
        try:
            self.llm_client = LLMClient(
                provider=llm_config.get("provider", "openrouter"),
                model=llm_config.get("model"),
                api_key=llm_config.get("api_key"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 8192)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}. Agent will run in limited mode.")
            self.llm_client = None
        
        # Initialize Web Search client
        web_config = config.get("web_search", {})
        try:
            self.web_search_client = WebSearchClient(web_config)
        except Exception as e:
            logger.warning(f"Failed to initialize Web Search client: {e}. Web search will be disabled.")
            self.web_search_client = None
        
        # Initialize Embedding client
        embedding_config = config.get("embedding", {})
        try:
            self.embedding_client = EmbeddingClient(
                api_key=embedding_config.get("api_key"),
                model=embedding_config.get("model", "text-embedding-3-small")
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Embedding client: {e}. Proximity agent will use LLM fallback.")
            self.embedding_client = None
        
        # Initialize Rosetta task manager
        rosetta_config = config.get("rosetta", {})
        try:
            self.rosetta_manager = RosettaTaskManager(
                max_concurrent=rosetta_config.get("max_concurrent", 2),
                python_path=rosetta_config.get("python_path", "python3"),
                timeout=rosetta_config.get("timeout", 3600)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Rosetta manager: {e}. Rosetta validation will be disabled.")
            self.rosetta_manager = None
        
        # Initialize specialized agents
        self.generation_agent = GenerationAgent(self.memory, config.get("generation", {}), self.llm_client, self.web_search_client)
        self.reflection_agent = ReflectionAgent(self.memory, config.get("reflection", {}), self.llm_client, self.web_search_client)
        self.ranking_agent = RankingAgent(self.memory, config.get("ranking", {}), self.llm_client, self.web_search_client)
        self.proximity_agent = ProximityAgent(self.memory, config.get("proximity", {}), self.llm_client, self.web_search_client, self.embedding_client)
        self.evolution_agent = EvolutionAgent(self.memory, config.get("evolution", {}), self.llm_client, self.web_search_client)
        self.meta_review_agent = MetaReviewAgent(self.memory, config.get("meta_review", {}), self.llm_client, self.web_search_client)
        
        self.logger = logging.getLogger("SupervisorAgent")
        self.current_iteration = 0
        
        self.logger.info("SupervisorAgent initialized")
    
    async def run_research_cycle(
        self,
        research_goal: ResearchGoal,
        max_iterations: int = 10,
        hypotheses_per_iteration: int = 10
    ) -> Dict[str, Any]:
        """
        Run a complete research cycle.
        
        Workflow:
        1. Generation: Generate initial hypotheses
        2. Reflection: Review hypotheses (initial, full, deep)
        3. Ranking: Conduct tournament to rank hypotheses
        4. Evolution: Improve top-ranked hypotheses
        5. Meta-review: Synthesize insights and generate overview
        6. Repeat for max_iterations
        """
        self.logger.info(f"Starting research cycle for: {research_goal.description}")
        self.logger.info(f"Max iterations: {max_iterations}")
        
        # Use web search client as async context manager if available
        if self.web_search_client:
            async with self.web_search_client:
                return await self._run_research_cycle_impl(research_goal, max_iterations, hypotheses_per_iteration)
        else:
            return await self._run_research_cycle_impl(research_goal, max_iterations, hypotheses_per_iteration)
    
    async def _run_research_cycle_impl(
        self,
        research_goal: ResearchGoal,
        max_iterations: int,
        hypotheses_per_iteration: int
    ) -> Dict[str, Any]:
        """Internal implementation of research cycle"""
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration + 1
            self.logger.info(f"{'='*80}")
            self.logger.info(f"ITERATION {self.current_iteration}/{max_iterations}")
            self.logger.info(f"{'='*80}\n")
            
            # Step 1: Generation
            self.logger.info("="*80)
            self.logger.info("STEP 1: HYPOTHESIS GENERATION")
            self.logger.info("="*80)
            existing_hypotheses = list(self.memory.hypotheses.values())
            meta_review = self.memory.get_latest_meta_review()
            
            self.logger.info("🧬 Generation Agent")
            self.logger.info(f"  📋 Research Goal: {research_goal.description[:80]}...")
            self.logger.info(f"  📊 Existing Hypotheses: {len(existing_hypotheses)}")
            self.logger.info(f"  📝 Meta Review: {'Available' if meta_review else 'Not available'}")
            self.logger.info("-" * 80)
            
            gen_result = await self.generation_agent.run({
                "research_goal": research_goal,
                "existing_hypotheses": existing_hypotheses,
                "meta_review_feedback": meta_review
            })
            
            new_hypotheses = gen_result["hypotheses"]
            
            self.logger.info(f"✅ Generated {len(new_hypotheses)} new hypotheses")
            for i, hyp in enumerate(new_hypotheses, 1):
                preview = hyp.content[:120] + "..." if len(hyp.content) > 120 else hyp.content
                self.logger.info(f"  [{i}] {hyp.id}: {preview}")
            
            # Step 2: Reflection (Review)
            self.logger.info("="*80)
            self.logger.info("STEP 2: HYPOTHESIS REVIEW")
            self.logger.info("="*80)
            self.logger.info("🔍 Reflection Agent")
            self.logger.info(f"  📋 Reviewing: {len(new_hypotheses)} new hypotheses")
            self.logger.info("-" * 80)
            
            # 병렬화: 모든 가설의 initial review를 동시에 실행
            # Rosetta tasks tracking
            rosetta_tasks = {}  # hypothesis_id -> task
            
            async def review_hypothesis_parallel(hyp, idx):
                """Single hypothesis review workflow"""
                self.logger.info(f"[{idx}/{len(new_hypotheses)}] {hyp.id}")
                preview = hyp.content[:100] + "..." if len(hyp.content) > 100 else hyp.content
                self.logger.info(f"      {preview}")
                
                # Initial review
                initial_result = await self.reflection_agent.run({
                    "hypothesis": hyp,
                    "review_type": "initial"
                })
                
                if not initial_result["pass"]:
                    reason = initial_result.get('reason', 'N/A')
                    self.logger.info(f"      Initial: ❌ FAILED - {reason}")
                    self.logger.info(f"      Status: REJECTED")
                    hyp.status = HypothesisStatus.REJECTED
                    return None
                
                self.logger.info(f"      Initial: ✅ PASSED")
                
                # Full review
                full_result = await self.reflection_agent.run({
                    "hypothesis": hyp,
                    "review_type": "full"
                })
                
                if not full_result["pass"]:
                    reason = full_result.get('reason', 'N/A')
                    self.logger.info(f"      Full: ❌ FAILED - {reason}")
                    self.logger.info(f"      Status: REJECTED")
                    hyp.status = HypothesisStatus.REJECTED
                    return None
                
                self.logger.info(f"      Full: ✅ PASSED")
                
                # Deep verification for promising hypotheses
                if iteration > 0:  # Skip in first iteration
                    deep_result = await self.reflection_agent.run({
                        "hypothesis": hyp,
                        "review_type": "deep_verification"
                    })
                    
                    if not deep_result["pass"]:
                        reason = deep_result.get('reason', 'N/A')
                        self.logger.info(f"      Deep: ❌ FAILED - {reason}")
                        self.logger.info(f"      Status: REJECTED")
                        hyp.status = HypothesisStatus.REJECTED
                        return None
                    
                    self.logger.info(f"      Deep: ✅ PASSED")
                    
                    # Start Rosetta simulation in background immediately after deep review passes
                    if self.rosetta_manager:
                        self.logger.info(f"      🧬 Starting Rosetta simulation in background...")
                        task = await self.rosetta_manager.submit_task(
                            hypothesis_id=hyp.id,
                            hypothesis_content=hyp.content,
                            task_type='binding'
                        )
                        rosetta_tasks[hyp.id] = task
                        self.logger.info(f"      🧬 Rosetta task submitted: {hyp.id}")
                
                hyp.status = HypothesisStatus.IN_TOURNAMENT
                return hyp
            
            # 병렬로 모든 가설 리뷰 실행
            review_tasks = [
                review_hypothesis_parallel(hyp, idx + 1)
                for idx, hyp in enumerate(new_hypotheses)
            ]
            review_results = await asyncio.gather(*review_tasks)
            
            # None이 아닌 결과만 필터링 (통과한 가설들)
            reviewed_hypotheses = [hyp for hyp in review_results if hyp is not None]
            
            self.logger.info(f"✅ {len(reviewed_hypotheses)} hypotheses passed review")
            self.logger.info(f"🧬 {len(rosetta_tasks)} Rosetta tasks running in background")
            
            if len(reviewed_hypotheses) == 0:
                self.logger.warning("No hypotheses passed review. Continuing to next iteration.")
                continue
            
            # Step 3: Proximity Analysis
            self.logger.info("="*80)
            self.logger.info("STEP 3: PROXIMITY ANALYSIS")
            self.logger.info("="*80)
            self.logger.info("🔗 Proximity Agent")
            self.logger.info(f"  📋 Analyzing: {len(reviewed_hypotheses)} hypotheses")
            self.logger.info("-" * 80)
            
            prox_result = await self.proximity_agent.run({
                "hypotheses": reviewed_hypotheses,
                "research_goal": research_goal
            })
            proximity_graph = prox_result["proximity_graph"]
            
            self.logger.info(f"✅ Proximity graph: {len(proximity_graph)} nodes")
            
            # Step 4: Ranking Tournament
            self.logger.info("="*80)
            self.logger.info("STEP 4: RANKING TOURNAMENT")
            self.logger.info("="*80)
            self.logger.info("🏆 Ranking Agent")
            self.logger.info(f"  📋 Tournament size: {len(reviewed_hypotheses)} hypotheses")
            self.logger.info("-" * 80)
            
            rank_result = await self.ranking_agent.run({
                "hypotheses": reviewed_hypotheses,
                "proximity_graph": proximity_graph
            })
            
            rankings = rank_result["rankings"]
            self.logger.info(f"✅ Tournament complete")
            for i, rank in enumerate(rankings[:5], 1):
                hyp_id = rank['hypothesis_id']
                rating = rank.get('rating', 'N/A')
                self.logger.info(f"  [{i}] {hyp_id} (rating: {rating})")
            
            # Step 5: Evolution (if not first iteration)
            if iteration > 0:
                self.logger.info("="*80)
                self.logger.info("STEP 5: HYPOTHESIS EVOLUTION")
                self.logger.info("="*80)
                self.logger.info("🧬 Evolution Agent")
                top_hypotheses = self.memory.get_top_hypotheses(n=5)
                self.logger.info(f"  📋 Evolving: top {len(top_hypotheses)} hypotheses")
                self.logger.info("-" * 80)
                
                evol_result = await self.evolution_agent.run({
                    "top_hypotheses": top_hypotheses,
                    "method": "all"
                })
                
                evolved = evol_result["hypotheses"] 
                self.logger.info(f"✅ Evolved {len(evolved)} hypotheses")
                for i, hyp in enumerate(evolved, 1):
                    preview = hyp.content[:100] + "..." if len(hyp.content) > 100 else hyp.content
                    self.logger.info(f"  [{i}] {hyp.id}: {preview}")
            
            # Step 6: Meta-review
            self.logger.info("="*80)
            self.logger.info("STEP 6: META-REVIEW")
            self.logger.info("="*80)
            self.logger.info("📋 Meta-Review Agent")
            self.logger.info(f"  📋 Synthesizing insights from iteration {iteration+1}")
            self.logger.info("-" * 80)
            
            meta_result = await self.meta_review_agent.run({
                "type": "meta_review"
            })
            
            patterns_count = len(meta_result.get('meta_review', {}).get('patterns', []))
            self.logger.info(f"✅ Meta-review: {patterns_count} patterns identified")
            
            # Process completed Rosetta results and run observation review
            if self.rosetta_manager:
                completed_results = await self.rosetta_manager.get_completed_tasks()
                
                if completed_results:
                    self.logger.info("="*80)
                    self.logger.info("ROSETTA OBSERVATION REVIEW")
                    self.logger.info("="*80)
                    self.logger.info(f"🔬 Processing {len(completed_results)} completed Rosetta simulations")
                    self.logger.info("-" * 80)
                    
                    for hyp_id, result in completed_results.items():
                        # Get hypothesis from memory
                        hyp = self.memory.hypotheses.get(hyp_id)
                        if not hyp:
                            self.logger.warning(f"Hypothesis {hyp_id} not found in memory")
                            continue
                        
                        self.logger.info(f"🔬 [{hyp_id}] Running observation review with Rosetta results...")
                        
                        # Run observation review with Rosetta results
                        obs_result = await self.reflection_agent.run({
                            "hypothesis": hyp,
                            "review_type": "observation",
                            "rosetta_results": result
                        })
                        
                        if obs_result["pass"]:
                            self.logger.info(f"      Observation: ✅ PASSED (Rosetta validated)")
                            # Boost hypothesis rating for passing Rosetta validation
                            if hasattr(hyp, 'elo_rating'):
                                hyp.elo_rating += 50  # Bonus for experimental validation
                        else:
                            reason = obs_result.get('reason', 'N/A')
                            self.logger.info(f"      Observation: ❌ FAILED - {reason}")
                            hyp.status = HypothesisStatus.REJECTED
                            self.logger.info(f"      Status: REJECTED (Rosetta failed)")
            
            # Log iteration summary
            self.log_iteration_summary(iteration + 1, rankings)
        
        # Wait for all remaining Rosetta tasks to complete before generating final overview
        if self.rosetta_manager:
            self.logger.info("="*80)
            self.logger.info("WAITING FOR REMAINING ROSETTA TASKS")
            self.logger.info("="*80)
            self.logger.info("⏳ Waiting for all Rosetta simulations to complete...")
            self.logger.info("-" * 80)
            
            remaining_results = await self.rosetta_manager.wait_all(timeout=7200, check_interval=30)
            
            if remaining_results:
                self.logger.info(f"✅ {len(remaining_results)} Rosetta tasks completed")
                
                # Run observation review for remaining results
                for hyp_id, result in remaining_results.items():
                    hyp = self.memory.hypotheses.get(hyp_id)
                    if not hyp:
                        continue
                    
                    self.logger.info(f"🔬 [{hyp_id}] Running final observation review...")
                    
                    obs_result = await self.reflection_agent.run({
                        "hypothesis": hyp,
                        "review_type": "observation",
                        "rosetta_results": result
                    })
                    
                    if obs_result["pass"]:
                        self.logger.info(f"      Observation: ✅ PASSED (Rosetta validated)")
                        if hasattr(hyp, 'elo_rating'):
                            hyp.elo_rating += 50
                    else:
                        reason = obs_result.get('reason', 'N/A')
                        self.logger.info(f"      Observation: ❌ FAILED - {reason}")
                        hyp.status = HypothesisStatus.REJECTED
        
        # Generate final research overview
        self.logger.info("="*80)
        self.logger.info("FINAL RESEARCH OVERVIEW")
        self.logger.info("="*80)
        self.logger.info("📊 Generating final summary...")
        self.logger.info("-" * 80)
        
        overview_result = await self.meta_review_agent.run({
            "type": "research_overview",
            "research_goal": research_goal
        })
        
        return {
            "status": "success",
            "iterations_completed": max_iterations,
            "total_hypotheses": len(self.memory.hypotheses),
            "top_hypotheses": self.memory.get_top_hypotheses(n=10),
            "research_overview": overview_result["overview"]
        }
    
    def log_iteration_summary(self, iteration: int, rankings: List[Dict]) -> None:
        """Log summary of iteration"""
        self.logger.info(f"--- Iteration {iteration} Summary ---")
        self.logger.info(f"Total hypotheses in memory: {len(self.memory.hypotheses)}")
        self.logger.info(f"Total reviews: {len(self.memory.reviews)}")
        self.logger.info(f"Total tournament matches: {len(self.memory.tournament_matches)}")
        
        self.logger.info("Top 5 Hypotheses:")
        for i, rank_info in enumerate(rankings[:5], 1):
            self.logger.info(
                f"  {i}. [{rank_info['hypothesis_id']}] "
                f"Elo: {rank_info['elo_rating']:.1f} | "
                f"W/L: {rank_info['wins']}/{rank_info['losses']}"
            )