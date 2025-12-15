"""
Ranking Agent
"""

import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import Hypothesis, Review, ResearchGoal, HypothesisStatus, TournamentMatch
from ..clients import LLMClient, WebSearchClient, EmbeddingClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """
    Evaluates and ranks hypotheses using Elo-based tournament.
    
    Features:
    - Pairwise comparisons with scientific debates
    - Multi-turn debates for top-ranked hypotheses
    - Single-turn comparisons for lower-ranked hypotheses
    - Optimized tournament matching based on proximity
    """
    
    def __init__(self, memory: ContextMemory, config: Dict[str, Any], llm_client: Optional[LLMClient] = None, web_search: Optional[WebSearchClient] = None):
        super().__init__("RankingAgent", memory, config, llm_client, web_search)
        self.k_factor = config.get("elo_k_factor", 32)
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run tournament to rank hypotheses"""
        hypotheses = task.get("hypotheses", [])
        proximity_graph = task.get("proximity_graph", {})
        
        self.log(f"Running tournament with {len(hypotheses)} hypotheses")
        
        # Organize tournament matches
        matches = self.organize_matches(hypotheses, proximity_graph)
        
        # Execute matches in parallel
        results = []
        
        # 병렬로 모든 매치 실행
        match_tasks = [
            self.conduct_match(
                match_config["hyp_a"],
                match_config["hyp_b"],
                match_config["debate_turns"]
            )
            for match_config in matches
        ]
        
        results = await asyncio.gather(*match_tasks)
        
        # Update Elo ratings for all matches
        for i, result in enumerate(results):
            match_config = matches[i]
            self.update_elo_ratings(
                match_config["hyp_a"],
                match_config["hyp_b"],
                result["winner_id"]
            )
        
        # Get updated rankings
        rankings = self.get_rankings(hypotheses)
        
        return {
            "status": "success",
            "matches_conducted": len(results),
            "rankings": rankings
        }
    
    def organize_matches(
        self,
        hypotheses: List[Hypothesis],
        proximity_graph: Dict
    ) -> List[Dict]:
        """Organize tournament matches with prioritization"""
        self.log("Organizing tournament matches")
        
        matches = []
        # Priority 1: Compare similar hypotheses
        # Priority 2: Compare new hypotheses
        # Priority 3: Compare top-ranked hypotheses
        
        # TODO: Implement sophisticated match organization
        # For now, simple pairwise matching
        for i in range(0, len(hypotheses) - 1, 2):
            is_top_tier = (hypotheses[i].elo_rating > 1400 and 
                          hypotheses[i+1].elo_rating > 1400)
            
            matches.append({
                "hyp_a": hypotheses[i],
                "hyp_b": hypotheses[i + 1],
                "debate_turns": 3 if is_top_tier else 1
            })
        
        return matches
    
    async def conduct_match(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        debate_turns: int
    ) -> Dict[str, Any]:
        """Conduct a tournament match between two hypotheses"""
        self.log(f"Match: {hyp_a.id} vs {hyp_b.id} ({debate_turns} turns)")
        
        if not self.llm:
            self.log("LLM not configured, using random winner", "warning")
            winner_id = hyp_a.id if hyp_a.elo_rating >= hyp_b.elo_rating else hyp_b.id
        else:
            try:
                # Conduct debate
                debate_history = []
                for round_num in range(1, debate_turns + 1):
                    prompt = self.prompt_manager.get_prompt(
                        "ranking_tournament_debate",
                        research_goal="Research goal context",  # Would get from task
                        hypothesis_a=f"{hyp_a.summary}\n\n{hyp_a.content}",
                        hypothesis_b=f"{hyp_b.summary}\n\n{hyp_b.content}",
                        round_num=round_num,
                        total_rounds=debate_turns,
                        previous_debate="\n".join([str(d) for d in debate_history])
                    )
                    
                    debate_response = await self.llm.generate_json(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        purpose="ranking tournament debate"
                    )
                    debate_history.append(debate_response)
                
                # Final decision
                final_prompt = self.prompt_manager.get_prompt(
                    "ranking_tournament_final",
                    research_goal="Research goal context",
                    hypothesis_a=hyp_a.content,
                    hypothesis_b=hyp_b.content,
                    elo_a=hyp_a.elo_rating,
                    elo_b=hyp_b.elo_rating,
                    debate_summary=json.dumps(debate_history, indent=2)
                )
                
                final_response = await self.llm.generate_json(
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.2,
                    purpose="ranking tournament decision"
                )
                
                # Parse winner with fallback
                winner_choice = final_response.get("winner", "").upper()
                if winner_choice == "A":
                    winner_id = hyp_a.id
                elif winner_choice == "B":
                    winner_id = hyp_b.id
                else:
                    # Fallback: use ELO rating
                    self.log(f"Invalid winner choice: {winner_choice}, using ELO fallback", "warning")
                    winner_id = hyp_a.id if hyp_a.elo_rating >= hyp_b.elo_rating else hyp_b.id
                
                decision_rationale = final_response.get("rationale", "No rationale provided")
                
            except Exception as e:
                self.log(f"Error in tournament match: {e}", "error")
                winner_id = hyp_a.id if hyp_a.elo_rating >= hyp_b.elo_rating else hyp_b.id
                decision_rationale = f"Error in debate, defaulting to higher Elo: {e}"
                debate_history = []
        
        match = TournamentMatch(
            match_id=f"match_{datetime.now().timestamp()}",
            hypothesis_a_id=hyp_a.id,
            hypothesis_b_id=hyp_b.id,
            timestamp=datetime.now(),
            debate_rounds=debate_history if 'debate_history' in locals() else [],
            winner_id=winner_id,
            decision_rationale=decision_rationale if 'decision_rationale' in locals() else "Default decision"
        )
        
        self.memory.store_tournament_match(match)
        
        return {
            "match_id": match.match_id,
            "winner_id": match.winner_id,
            "rationale": match.decision_rationale
        }
    
    def update_elo_ratings(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        winner_id: str
    ) -> None:
        """Update Elo ratings based on match result"""
        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((hyp_b.elo_rating - hyp_a.elo_rating) / 400))
        expected_b = 1 - expected_a
        
        # Actual scores
        score_a = 1.0 if winner_id == hyp_a.id else 0.0
        score_b = 1.0 - score_a
        
        # Update ratings
        hyp_a.elo_rating += self.k_factor * (score_a - expected_a)
        hyp_b.elo_rating += self.k_factor * (score_b - expected_b)
        
        # Update win/loss records
        if winner_id == hyp_a.id:
            hyp_a.wins += 1
            hyp_b.losses += 1
        else:
            hyp_b.wins += 1
            hyp_a.losses += 1
        
        self.log(f"Updated ratings: {hyp_a.id}={hyp_a.elo_rating:.1f}, "
                f"{hyp_b.id}={hyp_b.elo_rating:.1f}")
    
    def get_rankings(self, hypotheses: List[Hypothesis]) -> List[Dict]:
        """Get current rankings of all hypotheses"""
        sorted_hyps = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)
        
        rankings = []
        for rank, hyp in enumerate(sorted_hyps, 1):
            rankings.append({
                "rank": rank,
                "hypothesis_id": hyp.id,
                "elo_rating": hyp.elo_rating,
                "wins": hyp.wins,
                "losses": hyp.losses,
                "summary": hyp.summary
            })
        
        return rankings

