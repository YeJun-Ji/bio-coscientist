"""
Tournament Ranking Agent - v6.0 Win-Count Based Pipeline

This agent ranks RequirementAnswers using tournament-style pairwise comparison.
NO ELO rating - uses pure win count for ranking.

Tournament Structure:
- 3 answers: Round-robin (A vs B, B vs C, A vs C) → Rank by win count
- 2 answers: Single matchup (A vs B)
- 1 answer: Auto-select (no tournament)

Tie-breaking:
- When answers have equal wins, LLM determines which is more novel/creative
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core import RequirementAnswer
from ..core.evaluation_results import Matchup, TournamentResult
from ..external_apis import LLMClient
from ..memory import ContextMemory
from ..prompts import PromptManager
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TournamentRankingAgent(BaseAgent):
    """
    Ranks RequirementAnswers using LLM-based tournament comparison.

    Architecture Note (v6.0):
    - Uses pure win count for ranking (NO ELO)
    - Tie-breaking via LLM novelty evaluation
    - Runs in parallel with Reflection Agent
    """

    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        **kwargs
    ):
        super().__init__(
            name="tournament_ranking",
            memory=memory,
            config=config,
            llm_client=llm_client,
            **kwargs
        )
        self.prompt_manager = prompt_manager or PromptManager()
        self.log("TournamentRankingAgent initialized (v6.0 - win-count based, novelty tiebreaker)")

    # ========================================================================
    # Main Entry Point (Required by BaseAgent)
    # ========================================================================

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (required by BaseAgent).
        Delegates to run_tournament.
        """
        answers = task.get("answers", [])
        requirement = task.get("requirement", {})
        return await self.run_tournament(answers=answers, requirement=requirement)

    # ========================================================================
    # Tournament Logic
    # ========================================================================

    async def run_tournament(
        self,
        answers: List[RequirementAnswer],
        requirement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run tournament-style ranking.

        Args:
            answers: List of RequirementAnswers to rank
            requirement: Requirement specification

        Returns:
            {
                "ranked_answers": List[RequirementAnswer] (sorted by rank),
                "tournament_results": Dict with matchup details
            }

        Process:
            1. Determine bracket based on number of answers
            2. For each matchup: LLM pairwise comparison
            3. Aggregate results into final ranking
            4. Break ties using novelty evaluation
        """
        num_answers = len(answers)
        self.log(f"Starting tournament with {num_answers} answers")

        if num_answers == 0:
            self.log("No answers to rank", level="warning")
            return {"ranked_answers": [], "tournament_results": None}

        if num_answers == 1:
            # Auto-select single answer
            self.log("Single answer - auto-selecting")
            answers[0].tournament_rank = 1
            answers[0].tournament_matchups = []
            return {
                "ranked_answers": answers,
                "tournament_results": {"method": "auto_select", "matchups": []}
            }

        if num_answers == 2:
            # Single matchup
            self.log("Two answers - single matchup")
            matchup = await self._compare_pair(answers[0], answers[1], requirement)
            winner_id = matchup.winner_id
            winner = next(a for a in answers if self._get_answer_id(a) == winner_id)
            loser = next(a for a in answers if self._get_answer_id(a) != winner_id)

            winner.tournament_rank = 1
            winner.wins = 1
            winner.losses = 0
            winner.tournament_matchups = [matchup.to_dict()]

            loser.tournament_rank = 2
            loser.wins = 0
            loser.losses = 1
            loser.tournament_matchups = [matchup.to_dict()]

            return {
                "ranked_answers": [winner, loser],
                "tournament_results": {
                    "method": "single_matchup",
                    "matchups": [matchup.to_dict()]
                }
            }

        # 3+ answers: Round-robin tournament
        self.log(f"{num_answers} answers - running round-robin tournament")
        matchups = []
        win_counts = {self._get_answer_id(a): 0 for a in answers}
        loss_counts = {self._get_answer_id(a): 0 for a in answers}

        # All pairwise comparisons
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                matchup = await self._compare_pair(answers[i], answers[j], requirement)
                matchups.append(matchup)

                winner_id = matchup.winner_id
                loser_id = matchup.answer_a_id if matchup.answer_b_id == winner_id else matchup.answer_b_id

                win_counts[winner_id] += 1
                loss_counts[loser_id] += 1

                self.log(
                    f"  Matchup: {self._get_answer_id(answers[i])[:8]} vs {self._get_answer_id(answers[j])[:8]} "
                    f"→ Winner: {winner_id[:8]} (margin: {matchup.margin})"
                )

        # Sort by win count
        ranked_answers = sorted(answers, key=lambda a: win_counts[self._get_answer_id(a)], reverse=True)

        # Handle ties with novelty evaluation
        ranked_answers = await self._break_ties_with_novelty(
            ranked_answers, matchups, win_counts, requirement
        )

        # Update tournament metadata
        for rank, answer in enumerate(ranked_answers, 1):
            answer_id = self._get_answer_id(answer)
            answer.tournament_rank = rank
            answer.wins = win_counts[answer_id]
            answer.losses = loss_counts[answer_id]
            answer.tournament_matchups = [
                m.to_dict() for m in matchups
                if m.answer_a_id == answer_id or m.answer_b_id == answer_id
            ]

        self.log(
            f"Tournament complete. Rankings: "
            f"{[(self._get_answer_id(a)[:8], win_counts[self._get_answer_id(a)]) for a in ranked_answers]}"
        )

        return {
            "ranked_answers": ranked_answers,
            "tournament_results": {
                "method": "round_robin",
                "matchups": [m.to_dict() for m in matchups],
                "win_counts": win_counts
            }
        }

    async def _compare_pair(
        self,
        answer_a: RequirementAnswer,
        answer_b: RequirementAnswer,
        requirement: Dict[str, Any]
    ) -> Matchup:
        """
        Single pairwise comparison using LLM.

        Args:
            answer_a: First answer
            answer_b: Second answer
            requirement: Requirement specification

        Returns:
            Matchup with winner_id, reasoning, margin
        """
        # Build prompt
        prompt = self.prompt_manager.get_prompt(
            "ranking/tournament_comparison",
            requirement=requirement,
            answer_a=self._answer_to_dict(answer_a),
            answer_b=self._answer_to_dict(answer_b)
        )

        # Call LLM
        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistency
                purpose="tournament_pairwise_comparison"
            )

            answer_a_id = self._get_answer_id(answer_a)
            answer_b_id = self._get_answer_id(answer_b)
            winner_id_from_llm = response.get("winner_id", "")

            # Validate winner_id - must be one of the two answer IDs
            if winner_id_from_llm == answer_a_id:
                winner_id = answer_a_id
            elif winner_id_from_llm == answer_b_id:
                winner_id = answer_b_id
            elif answer_a_id in winner_id_from_llm or winner_id_from_llm in answer_a_id:
                # Partial match - LLM might have truncated the ID
                winner_id = answer_a_id
                self.log(f"  Warning: LLM returned partial match '{winner_id_from_llm}' → mapped to '{answer_a_id}'")
            elif answer_b_id in winner_id_from_llm or winner_id_from_llm in answer_b_id:
                winner_id = answer_b_id
                self.log(f"  Warning: LLM returned partial match '{winner_id_from_llm}' → mapped to '{answer_b_id}'")
            else:
                # Default to answer_a if LLM returned completely invalid ID
                winner_id = answer_a_id
                self.log(f"  Warning: Invalid winner_id '{winner_id_from_llm}' → defaulting to '{answer_a_id}'")

            return Matchup(
                answer_a_id=answer_a_id,
                answer_b_id=answer_b_id,
                winner_id=winner_id,
                reasoning=response.get("reasoning", "No reasoning provided"),
                margin=response.get("margin", "close")
            )

        except Exception as e:
            self.log(f"ERROR in pairwise comparison: {e}", level="error")
            # Fallback: Random selection (first answer)
            return Matchup(
                answer_a_id=self._get_answer_id(answer_a),
                answer_b_id=self._get_answer_id(answer_b),
                winner_id=self._get_answer_id(answer_a),
                reasoning=f"LLM comparison failed: {str(e)}. Defaulting to first answer.",
                margin="very_close"
            )

    async def _break_ties_with_novelty(
        self,
        ranked_answers: List[RequirementAnswer],
        matchups: List[Matchup],
        win_counts: Dict[str, int],
        requirement: Dict[str, Any]
    ) -> List[RequirementAnswer]:
        """
        Break ties using LLM novelty evaluation.

        When answers have equal wins, ask LLM which is more novel/creative.

        Args:
            ranked_answers: Answers sorted by win count
            matchups: All matchups
            win_counts: Win count per answer
            requirement: Requirement specification

        Returns:
            Re-sorted answers with ties broken by novelty
        """
        # Group by win count
        win_count_groups = {}
        for answer in ranked_answers:
            count = win_counts[self._get_answer_id(answer)]
            if count not in win_count_groups:
                win_count_groups[count] = []
            win_count_groups[count].append(answer)

        # For each group with ties, use novelty evaluation
        result = []
        for count in sorted(win_count_groups.keys(), reverse=True):
            group = win_count_groups[count]
            if len(group) == 1:
                result.extend(group)
            else:
                # Use LLM to determine novelty
                self.log(f"  Breaking tie for {len(group)} answers with {count} wins using novelty")
                sorted_group = await self._sort_by_novelty(group, requirement)
                result.extend(sorted_group)

        return result

    async def _sort_by_novelty(
        self,
        tied_answers: List[RequirementAnswer],
        requirement: Dict[str, Any]
    ) -> List[RequirementAnswer]:
        """
        Sort tied answers by novelty using LLM.

        Args:
            tied_answers: Answers with equal win counts
            requirement: Requirement specification

        Returns:
            Sorted list with most novel first
        """
        if len(tied_answers) <= 1:
            return tied_answers

        # Build prompt for novelty comparison
        prompt = self.prompt_manager.get_prompt(
            "ranking/novelty_tiebreaker",
            requirement=requirement,
            answers=[self._answer_to_dict(a) for a in tied_answers]
        )

        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                purpose="novelty_tiebreaker"
            )

            more_novel_id = response.get("more_novel_id")
            self.log(f"  Novelty winner: {more_novel_id[:8] if more_novel_id else 'none'}")

            # Find the most novel answer and put it first
            novel_answer = None
            other_answers = []
            for answer in tied_answers:
                if self._get_answer_id(answer) == more_novel_id:
                    novel_answer = answer
                    # Mark as novelty winner
                    if hasattr(answer, 'metadata'):
                        answer.metadata["is_novelty_winner"] = True
                else:
                    other_answers.append(answer)

            if novel_answer:
                return [novel_answer] + other_answers
            else:
                return tied_answers

        except Exception as e:
            self.log(f"ERROR in novelty comparison: {e}", level="error")
            return tied_answers  # Keep original order on error

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_answer_id(self, answer) -> str:
        """Get answer ID supporting both dict and object forms."""
        if isinstance(answer, dict):
            return answer.get("id", "unknown")
        return getattr(answer, "id", "unknown")

    def _answer_to_dict(self, answer) -> Dict[str, Any]:
        """Convert answer to dict for prompt template."""
        if isinstance(answer, dict):
            return answer

        return {
            "id": getattr(answer, "id", "unknown"),
            "answer": getattr(answer, "answer", ""),
            "rationale": getattr(answer, "rationale", ""),
            "deliverables": getattr(answer, "deliverables", {}),
            "metadata": getattr(answer, "metadata", {})
        }
