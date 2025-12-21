"""
Tournament Ranking Agent - v5.0 Simplified Evaluation Pipeline

This agent ranks RequirementAnswers using tournament-style pairwise comparison.

Tournament Structure:
- 3 answers: Round-robin (A vs B, B vs C, A vs C) → Rank by win count
- 2 answers: Single matchup (A vs B)
- 1 answer: Auto-select (no tournament)

Comparison Criteria (via LLM):
- Goal Contribution (40%): Which helps next requirement more?
- Data Confidence (30%): Which uses more authoritative sources?
- Feasibility & Cost (30%): Which is cheaper/faster to validate?
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

    Architecture Note:
    - Replaces score-based ranking with pairwise LLM comparison
    - Uses win count for final ranking
    - Updates ELO ratings for backward compatibility
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
        self.log("TournamentRankingAgent initialized (v5.0 - tournament-based)")

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
            4. Update ELO ratings for backward compatibility
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
            answers[0].elo_rating = 1200
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
            winner = next(a for a in answers if a.id == winner_id)
            loser = next(a for a in answers if a.id != winner_id)

            winner.tournament_rank = 1
            winner.elo_rating = 1250
            winner.tournament_matchups = [matchup.to_dict()]

            loser.tournament_rank = 2
            loser.elo_rating = 1150
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
        win_counts = {a.id: 0 for a in answers}

        # All pairwise comparisons
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                matchup = await self._compare_pair(answers[i], answers[j], requirement)
                matchups.append(matchup)
                win_counts[matchup.winner_id] += 1
                self.log(
                    f"  Matchup: {answers[i].id[:8]} vs {answers[j].id[:8]} "
                    f"→ Winner: {matchup.winner_id[:8]} (margin: {matchup.margin})"
                )

        # Sort by win count
        ranked_answers = sorted(answers, key=lambda a: win_counts[a.id], reverse=True)

        # Handle ties
        ranked_answers = self._break_ties(ranked_answers, matchups, win_counts)

        # Update ELO ratings and tournament metadata
        for rank, answer in enumerate(ranked_answers, 1):
            answer.tournament_rank = rank
            answer.elo_rating = 1200 + (len(ranked_answers) - rank) * 50
            answer.tournament_matchups = [
                m.to_dict() for m in matchups
                if m.answer_a_id == answer.id or m.answer_b_id == answer.id
            ]

        self.log(
            f"Tournament complete. Rankings: "
            f"{[(a.id[:8], win_counts[a.id]) for a in ranked_answers]}"
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
            Matchup with winner_id, reasoning, criteria_scores, margin
        """
        # Build prompt
        prompt = self.prompt_manager.get_prompt(
            "ranking/tournament_comparison",
            requirement=requirement,
            answer_a={
                "id": answer_a.id,
                "answer": answer_a.answer,
                "rationale": answer_a.rationale,
                "deliverables": answer_a.deliverables,
                "reflection_score": answer_a.quality_score  # From Reflection Agent
            },
            answer_b={
                "id": answer_b.id,
                "answer": answer_b.answer,
                "rationale": answer_b.rationale,
                "deliverables": answer_b.deliverables,
                "reflection_score": answer_b.quality_score
            }
        )

        # Call LLM
        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistency
                purpose="tournament_pairwise_comparison"
            )

            return Matchup(
                answer_a_id=answer_a.id,
                answer_b_id=answer_b.id,
                winner_id=response["winner_id"],
                reasoning=response["reasoning"],
                criteria_scores=response.get("criteria_scores", {}),
                margin=response.get("margin", "close")
            )

        except Exception as e:
            self.log(f"ERROR in pairwise comparison: {e}", level="error")
            # Fallback: Use reflection score
            winner_id = answer_a.id if answer_a.quality_score > answer_b.quality_score else answer_b.id
            return Matchup(
                answer_a_id=answer_a.id,
                answer_b_id=answer_b.id,
                winner_id=winner_id,
                reasoning=f"LLM comparison failed, using reflection scores (A: {answer_a.quality_score:.2f}, B: {answer_b.quality_score:.2f})",
                criteria_scores={},
                margin="close"
            )

    def _break_ties(
        self,
        ranked_answers: List[RequirementAnswer],
        matchups: List[Matchup],
        win_counts: Dict[str, int]
    ) -> List[RequirementAnswer]:
        """
        Break ties using margin of victory.

        Args:
            ranked_answers: Answers sorted by win count
            matchups: All matchups
            win_counts: Win count per answer

        Returns:
            Re-sorted answers with ties broken
        """
        # Group by win count
        win_count_groups = {}
        for answer in ranked_answers:
            count = win_counts[answer.id]
            if count not in win_count_groups:
                win_count_groups[count] = []
            win_count_groups[count].append(answer)

        # For each group with ties, use margin of victory
        result = []
        for count in sorted(win_count_groups.keys(), reverse=True):
            group = win_count_groups[count]
            if len(group) == 1:
                result.extend(group)
            else:
                # Calculate total margin score
                margin_scores = {"clear": 2, "close": 1, "very_close": 0.5}
                margins = {}
                for answer in group:
                    total_margin = 0
                    for matchup in matchups:
                        if matchup.winner_id == answer.id:
                            total_margin += margin_scores.get(matchup.margin, 1)
                    margins[answer.id] = total_margin

                # Sort by margin
                sorted_group = sorted(group, key=lambda a: margins[a.id], reverse=True)
                result.extend(sorted_group)
                self.log(f"  Tie broken for {len(group)} answers with {count} wins using margin")

        return result
