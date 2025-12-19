"""
Ranking Agent - RequirementAnswer Tournament System

This agent ranks RequirementAnswers using ELO-based tournament
for the Sequential Confirmation workflow.
"""

import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import ResearchGoal, RequirementAnswer
from ..external_apis import LLMClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """
    Ranks RequirementAnswers using ELO-based tournament system.

    Features:
    - Pairwise comparisons for answers within same requirement
    - ELO rating updates after each match
    - Multi-turn debates for high-rated answers
    """

    def __init__(self, memory: ContextMemory, config: Dict[str, Any], llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(name="ranking", memory=memory, config=config, llm_client=llm_client, **kwargs)
        # Read elo_k_factor from execution_plan (v3.0 config)
        agent_cfg = config.get("execution_plan", {}).get("agent_config", {}).get("ranking", {})
        self.k_factor = agent_cfg.get("elo_k_factor", 32)
        # Track ELO changes during this run for memory synchronization
        self._elo_changes: Dict[str, float] = {}
        # Track wins/losses changes during this run for memory synchronization
        self._win_loss_changes: Dict[str, Dict[str, int]] = {}

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

    def _get_answer_elo(self, answer) -> float:
        """Get answer ELO from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("elo_rating", 1200.0)
        return getattr(answer, "elo_rating", 1200.0)

    def _set_answer_elo(self, answer, value: float) -> None:
        """Set answer ELO for both dict and RequirementAnswer object"""
        if isinstance(answer, dict):
            answer["elo_rating"] = value
        else:
            answer.elo_rating = value

    def _get_answer_wins(self, answer) -> int:
        """Get answer wins from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("wins", 0)
        return getattr(answer, "wins", 0)

    def _set_answer_wins(self, answer, value: int) -> None:
        """Set answer wins for both dict and RequirementAnswer object"""
        if isinstance(answer, dict):
            answer["wins"] = value
        else:
            answer.wins = value

    def _get_answer_losses(self, answer) -> int:
        """Get answer losses from dict or RequirementAnswer object"""
        if isinstance(answer, dict):
            return answer.get("losses", 0)
        return getattr(answer, "losses", 0)

    def _set_answer_losses(self, answer, value: int) -> None:
        """Set answer losses for both dict and RequirementAnswer object"""
        if isinstance(answer, dict):
            answer["losses"] = value
        else:
            answer.losses = value

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
        Delegates to run_for_answers for RequirementAnswer ranking.
        """
        return await self.run_for_answers(task)

    async def run_for_answers(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run tournament to rank RequirementAnswers for the SAME requirement.

        This is the main entry point for Sequential Confirmation.
        Only answers for the same requirement can be compared.

        Args:
            task: {
                "requirement_id": str,
                "answers": List[RequirementAnswer],
                "requirement": Dict (optional, for context)
            }

        Returns:
            {
                "status": "success" | "error",
                "requirement_id": str,
                "matches_conducted": int,
                "rankings": List[Dict],
                "elo_updates": Dict[str, float]
            }
        """
        import time
        func_start = time.time()

        requirement_id = task.get("requirement_id", "")
        answers = task.get("answers", [])
        requirement = task.get("requirement", {})

        self.log(f"[ANSWER-RANKING] Running tournament for requirement {requirement_id}")
        self.log(f"[ANSWER-RANKING] {len(answers)} answers to compare")

        if len(answers) < 2:
            self.log("[ANSWER-RANKING] Not enough answers for tournament")
            return {
                "status": "success",
                "requirement_id": requirement_id,
                "matches_conducted": 0,
                "rankings": self._get_answer_rankings(answers),
                "elo_updates": {}
            }

        # Clear changes from previous run
        self._elo_changes = {}
        self._win_loss_changes = {}

        try:
            # Organize matches
            matches = self._organize_answer_matches(answers)
            self.log(f"[ANSWER-RANKING] Organized {len(matches)} matches")

            # Execute matches in parallel
            match_tasks = [
                self._conduct_answer_match(
                    match_config["answer_a"],
                    match_config["answer_b"],
                    requirement,
                    match_config["debate_turns"]
                )
                for match_config in matches
            ]

            results = await asyncio.gather(*match_tasks)

            # Update ELO ratings
            for i, result in enumerate(results):
                match_config = matches[i]
                self._update_answer_elo_ratings(
                    match_config["answer_a"],
                    match_config["answer_b"],
                    result["winner_id"]
                )

            # Get updated rankings
            rankings = self._get_answer_rankings(answers)

            # Update status to "ranked" for all answers
            for answer in answers:
                if isinstance(answer, dict):
                    answer["status"] = "ranked"
                else:
                    answer.mark_ranked()

            func_duration = time.time() - func_start
            self.log(f"[ANSWER-RANKING] ✓ Complete ({func_duration:.2f}s)")
            self.log(f"[ANSWER-RANKING] {len(results)} matches, top answer: {rankings[0]['id'] if rankings else 'N/A'}")

            return {
                "status": "success",
                "requirement_id": requirement_id,
                "matches_conducted": len(results),
                "rankings": rankings,
                "elo_updates": dict(self._elo_changes),
                "win_loss_updates": dict(self._win_loss_changes)
            }

        except Exception as e:
            self.log(f"[ANSWER-RANKING] ✗ Error: {e}", "error")
            import traceback
            self.log(f"[ANSWER-RANKING] {traceback.format_exc()}", "debug")
            return {
                "status": "error",
                "requirement_id": requirement_id,
                "message": str(e)
            }

    def _organize_answer_matches(self, answers: List) -> List[Dict]:
        """Organize tournament matches for answers"""
        matches = []

        # Sort by ELO for better matchups
        sorted_answers = sorted(
            answers,
            key=lambda a: self._get_answer_elo(a),
            reverse=True
        )

        # Pairwise matching
        for i in range(0, len(sorted_answers) - 1, 2):
            elo_a = self._get_answer_elo(sorted_answers[i])
            elo_b = self._get_answer_elo(sorted_answers[i + 1])
            is_top_tier = (elo_a > 1400 and elo_b > 1400)

            matches.append({
                "answer_a": sorted_answers[i],
                "answer_b": sorted_answers[i + 1],
                "debate_turns": 2 if is_top_tier else 1  # Fewer turns for answers
            })

        return matches

    async def _conduct_answer_match(
        self,
        answer_a,
        answer_b,
        requirement: Dict[str, Any],
        debate_turns: int
    ) -> Dict[str, Any]:
        """Conduct a tournament match between two answers"""
        answer_a_id = self._get_answer_id(answer_a)
        answer_b_id = self._get_answer_id(answer_b)

        self.log(f"  Match: {answer_a_id} vs {answer_b_id}")

        if not self.llm:
            # Fallback: use ELO
            elo_a = self._get_answer_elo(answer_a)
            elo_b = self._get_answer_elo(answer_b)
            winner_id = answer_a_id if elo_a >= elo_b else answer_b_id
            return {"winner_id": winner_id, "rationale": "ELO fallback"}

        try:
            # Build answer comparison prompt
            prompt = self.prompt_manager.get_prompt(
                "ranking/answer_comparison",
                requirement=requirement,
                answer_a={
                    "id": answer_a_id,
                    "answer": self._get_answer_content(answer_a),
                    "rationale": self._get_answer_rationale(answer_a),
                    "deliverables": self._get_answer_deliverables(answer_a),
                    "elo_rating": self._get_answer_elo(answer_a)
                },
                answer_b={
                    "id": answer_b_id,
                    "answer": self._get_answer_content(answer_b),
                    "rationale": self._get_answer_rationale(answer_b),
                    "deliverables": self._get_answer_deliverables(answer_b),
                    "elo_rating": self._get_answer_elo(answer_b)
                }
            )

            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                purpose="answer_comparison"
            )

            winner_choice = response.get("winner", "").upper()
            if winner_choice == "A":
                winner_id = answer_a_id
            elif winner_choice == "B":
                winner_id = answer_b_id
            else:
                # Fallback
                elo_a = self._get_answer_elo(answer_a)
                elo_b = self._get_answer_elo(answer_b)
                winner_id = answer_a_id if elo_a >= elo_b else answer_b_id

            return {
                "winner_id": winner_id,
                "rationale": response.get("rationale", ""),
                "comparison": response
            }

        except Exception as e:
            self.log(f"  Error in match: {e}", "warning")
            elo_a = self._get_answer_elo(answer_a)
            elo_b = self._get_answer_elo(answer_b)
            winner_id = answer_a_id if elo_a >= elo_b else answer_b_id
            return {"winner_id": winner_id, "rationale": f"Error fallback: {e}"}

    def _update_answer_elo_ratings(self, answer_a, answer_b, winner_id: str) -> None:
        """Update ELO ratings for answers after a match"""
        answer_a_id = self._get_answer_id(answer_a)
        answer_b_id = self._get_answer_id(answer_b)

        elo_a = self._get_answer_elo(answer_a)
        elo_b = self._get_answer_elo(answer_b)

        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 - expected_a

        # Actual scores
        if winner_id == answer_a_id:
            actual_a, actual_b = 1, 0
        else:
            actual_a, actual_b = 0, 1

        # Update ELO
        new_elo_a = elo_a + self.k_factor * (actual_a - expected_a)
        new_elo_b = elo_b + self.k_factor * (actual_b - expected_b)

        self._set_answer_elo(answer_a, new_elo_a)
        self._set_answer_elo(answer_b, new_elo_b)

        # Track changes
        self._elo_changes[answer_a_id] = new_elo_a
        self._elo_changes[answer_b_id] = new_elo_b

        # Update wins/losses
        wins_a = self._get_answer_wins(answer_a)
        losses_a = self._get_answer_losses(answer_a)
        wins_b = self._get_answer_wins(answer_b)
        losses_b = self._get_answer_losses(answer_b)

        if winner_id == answer_a_id:
            self._set_answer_wins(answer_a, wins_a + 1)
            self._set_answer_losses(answer_b, losses_b + 1)
            self._win_loss_changes[answer_a_id] = {"wins": wins_a + 1, "losses": losses_a}
            self._win_loss_changes[answer_b_id] = {"wins": wins_b, "losses": losses_b + 1}
        else:
            self._set_answer_wins(answer_b, wins_b + 1)
            self._set_answer_losses(answer_a, losses_a + 1)
            self._win_loss_changes[answer_a_id] = {"wins": wins_a, "losses": losses_a + 1}
            self._win_loss_changes[answer_b_id] = {"wins": wins_b + 1, "losses": losses_b}

    def _get_answer_rankings(self, answers: List) -> List[Dict]:
        """Get current rankings for answers sorted by ELO"""
        ranked = sorted(
            answers,
            key=lambda a: self._get_answer_elo(a),
            reverse=True
        )

        return [
            {
                "rank": i + 1,
                "id": self._get_answer_id(a),
                "requirement_id": self._get_answer_requirement_id(a),
                "elo_rating": self._get_answer_elo(a),
                "wins": self._get_answer_wins(a),
                "losses": self._get_answer_losses(a)
            }
            for i, a in enumerate(ranked)
        ]
