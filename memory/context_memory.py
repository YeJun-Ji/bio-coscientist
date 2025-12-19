"""
Context Memory System - RequirementAnswer Storage for Sequential Confirmation

Maintains research context and stores RequirementAnswers for the
Sequential Confirmation workflow.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from ..core import ResearchGoal, RequirementAnswer

logger = logging.getLogger(__name__)


class ContextMemory:
    """
    Maintains research context and RequirementAnswer storage.

    Supports the Sequential Confirmation workflow:
    - Stores all generated RequirementAnswers
    - Tracks confirmed answers per requirement
    - Maintains version history for Speculative Exploration
    - Exports RA configs to logs/<session>/RAs/ folder
    """

    def __init__(self, storage_path: str = "./research_memory"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # RAs folder for config export (will be set by supervisor)
        self.ras_dir: Optional[str] = None

        # RequirementAnswer Storage (Sequential Confirmation)
        # All generated answers indexed by answer.id
        self.requirement_answers: Dict[str, RequirementAnswer] = {}
        # Confirmed answers indexed by requirement_id (only one per requirement)
        self.confirmed_answers: Dict[str, RequirementAnswer] = {}
        # Version history for Speculative Exploration
        self.confirmed_versions: Dict[str, List[RequirementAnswer]] = {}

        # Meta-level research context
        self.meta_reviews: List[Dict] = []
        self.research_overviews: List[Dict] = []

    def set_ras_directory(self, ras_dir: str) -> None:
        """Set the RAs directory for config export"""
        self.ras_dir = ras_dir
        os.makedirs(ras_dir, exist_ok=True)
        logger.info(f"RAs directory set: {ras_dir}")

    def get_research_goal(self) -> Optional[ResearchGoal]:
        """Get the current research goal"""
        return getattr(self, "_research_goal", None)

    def set_research_goal(self, goal: ResearchGoal):
        """Set the current research goal"""
        self._research_goal = goal

    def store_result(self, result_type: str, result: Any):
        """Store a generic result"""
        if not hasattr(self, "_results"):
            self._results = {}
        if result_type not in self._results:
            self._results[result_type] = []
        self._results[result_type].append(result)
        logger.info(f"Stored {result_type} result")

    def store_ranking(self, ranking_result: Dict):
        """Store ranking result"""
        self.store_result("ranking", ranking_result)

    def store_meta_review(self, meta_review: Dict) -> None:
        """Store meta-review insights"""
        self.meta_reviews.append(meta_review)
        logger.info(f"Stored meta-review iteration {len(self.meta_reviews)}")

    def get_latest_meta_review(self) -> Optional[Dict]:
        """Get the most recent meta-review"""
        return self.meta_reviews[-1] if self.meta_reviews else None

    def store_research_overview(self, overview: Dict) -> None:
        """Store research overview"""
        self.research_overviews.append(overview)
        logger.info(f"Stored research overview iteration {len(self.research_overviews)}")

    # ========== RequirementAnswer Methods (Sequential Confirmation) ==========

    def store_requirement_answer(self, answer: RequirementAnswer) -> None:
        """
        Store a RequirementAnswer in memory and export config to RAs folder.

        Args:
            answer: RequirementAnswer object to store
        """
        if not answer.id:
            logger.warning("Cannot store answer without ID")
            return

        self.requirement_answers[answer.id] = answer
        logger.info(f"Stored RequirementAnswer {answer.id} for requirement {answer.requirement_id}")

        # Export config to RAs folder if available
        if self.ras_dir:
            self._export_ra_config(answer)

    def get_requirement_answer(self, answer_id: str) -> Optional[RequirementAnswer]:
        """Retrieve a RequirementAnswer by ID"""
        return self.requirement_answers.get(answer_id)

    def get_answers_for_requirement(self, requirement_id: str) -> List[RequirementAnswer]:
        """
        Get all RequirementAnswers for a specific requirement.

        Args:
            requirement_id: The requirement ID to filter by

        Returns:
            List of RequirementAnswers for that requirement
        """
        return [
            answer for answer in self.requirement_answers.values()
            if answer.requirement_id == requirement_id
        ]

    def get_top_answers_for_requirement(
        self,
        requirement_id: str,
        n: int = 3,
        status_filter: Optional[str] = None
    ) -> List[RequirementAnswer]:
        """
        Get top N answers for a requirement, sorted by ELO.

        Args:
            requirement_id: The requirement to get answers for
            n: Number of top answers to return
            status_filter: Optional status filter (e.g., "reviewed", "ranked")

        Returns:
            List of top N RequirementAnswers sorted by ELO
        """
        answers = self.get_answers_for_requirement(requirement_id)

        if status_filter:
            answers = [a for a in answers if a.status == status_filter]

        sorted_answers = sorted(
            answers,
            key=lambda a: a.elo_rating,
            reverse=True
        )

        return sorted_answers[:n]

    def get_best_answer_per_requirement(self) -> Dict[str, RequirementAnswer]:
        """
        Get the best answer (highest ELO) for each requirement.

        Returns:
            Dict mapping requirement_id to best RequirementAnswer
        """
        best_answers = {}

        for answer in self.requirement_answers.values():
            req_id = answer.requirement_id
            if req_id not in best_answers:
                best_answers[req_id] = answer
            elif answer.elo_rating > best_answers[req_id].elo_rating:
                best_answers[req_id] = answer

        return best_answers

    def update_answer_elo(self, answer_id: str, new_elo: float) -> None:
        """
        Update ELO rating for a RequirementAnswer.

        Args:
            answer_id: ID of the answer to update
            new_elo: New ELO rating
        """
        if answer_id not in self.requirement_answers:
            logger.warning(f"Cannot update ELO: answer {answer_id} not found")
            return

        answer = self.requirement_answers[answer_id]
        old_elo = answer.elo_rating
        answer.elo_rating = new_elo
        logger.debug(f"Updated ELO for {answer_id}: {old_elo:.1f} → {new_elo:.1f}")

    def batch_update_answer_elo(self, elo_updates: Dict[str, float]) -> None:
        """
        Batch update ELO ratings for multiple answers.

        Args:
            elo_updates: Dict mapping answer_id to new ELO rating
        """
        if not elo_updates:
            return

        updated_count = 0
        for answer_id, new_elo in elo_updates.items():
            if answer_id in self.requirement_answers:
                self.update_answer_elo(answer_id, new_elo)
                updated_count += 1

        logger.info(f"Batch updated ELO for {updated_count}/{len(elo_updates)} answers")

    def update_answer_win_loss(
        self,
        answer_id: str,
        wins: int,
        losses: int
    ) -> None:
        """Update wins and losses for a RequirementAnswer"""
        if answer_id not in self.requirement_answers:
            logger.warning(f"Cannot update wins/losses: answer {answer_id} not found")
            return

        answer = self.requirement_answers[answer_id]
        answer.wins = wins
        answer.losses = losses

    def confirm_answer(self, answer_id: str) -> bool:
        """
        Confirm a RequirementAnswer as the final answer for its requirement.

        This is a key method in Sequential Confirmation:
        - Marks the answer as "confirmed"
        - Stores in confirmed_answers (one per requirement)
        - Adds to version history

        Args:
            answer_id: ID of the answer to confirm

        Returns:
            True if confirmation succeeded, False otherwise
        """
        if answer_id not in self.requirement_answers:
            logger.warning(f"Cannot confirm: answer {answer_id} not found")
            return False

        answer = self.requirement_answers[answer_id]
        req_id = answer.requirement_id

        # Update status
        answer.status = "confirmed"
        answer.mark_confirmed()

        # Store/replace confirmed answer for this requirement
        previous = self.confirmed_answers.get(req_id)
        self.confirmed_answers[req_id] = answer

        # Add to version history (for Speculative Exploration)
        if req_id not in self.confirmed_versions:
            self.confirmed_versions[req_id] = []
        self.confirmed_versions[req_id].append(answer)

        if previous:
            logger.info(f"✓ Replaced confirmed answer for {req_id}: {previous.id} → {answer.id}")
        else:
            logger.info(f"✓ Confirmed answer for {req_id}: {answer.id}")

        return True

    def get_confirmed_answer(self, requirement_id: str) -> Optional[RequirementAnswer]:
        """
        Get the confirmed answer for a specific requirement.

        Args:
            requirement_id: The requirement ID

        Returns:
            The confirmed RequirementAnswer, or None if not yet confirmed
        """
        return self.confirmed_answers.get(requirement_id)

    def get_all_confirmed_answers(self) -> Dict[str, RequirementAnswer]:
        """
        Get all confirmed answers.

        Returns:
            Dict mapping requirement_id to confirmed RequirementAnswer
        """
        return self.confirmed_answers.copy()

    def is_requirement_confirmed(self, requirement_id: str) -> bool:
        """
        Check if a requirement has been confirmed.

        Args:
            requirement_id: The requirement ID to check

        Returns:
            True if the requirement has a confirmed answer
        """
        return requirement_id in self.confirmed_answers

    def get_confirmed_versions(self, requirement_id: str) -> List[RequirementAnswer]:
        """
        Get all confirmed versions for a requirement (Speculative Exploration).

        Args:
            requirement_id: The requirement ID

        Returns:
            List of confirmed versions (ordered by confirmation time)
        """
        return self.confirmed_versions.get(requirement_id, [])

    def get_context_for_requirement(
        self,
        requirement_id: str,
        depends_on: List[str]
    ) -> Dict[str, RequirementAnswer]:
        """
        Get confirmed answers for dependencies of a requirement.

        This is used when generating/evolving answers for a requirement:
        the confirmed answers from its dependencies provide context.

        Args:
            requirement_id: The requirement being processed (for logging)
            depends_on: List of requirement IDs this one depends on

        Returns:
            Dict mapping dependency requirement_id to confirmed RequirementAnswer
        """
        context = {}
        for dep_id in depends_on:
            confirmed = self.get_confirmed_answer(dep_id)
            if confirmed:
                context[dep_id] = confirmed
            else:
                logger.warning(f"Dependency {dep_id} not confirmed for {requirement_id}")
        return context

    def get_answers_by_status(
        self,
        requirement_id: str,
        status: str
    ) -> List[RequirementAnswer]:
        """
        Get answers for a requirement filtered by status.

        Args:
            requirement_id: The requirement ID
            status: Status to filter by ("generated", "reviewed", "ranked", "confirmed")

        Returns:
            List of answers matching the status
        """
        return [
            answer for answer in self.requirement_answers.values()
            if answer.requirement_id == requirement_id and answer.status == status
        ]

    def get_unreviewed_answers(self, requirement_id: str) -> List[RequirementAnswer]:
        """Get answers that haven't been reviewed yet"""
        return self.get_answers_by_status(requirement_id, "generated")

    def get_reviewed_answers(self, requirement_id: str) -> List[RequirementAnswer]:
        """Get answers that have been reviewed but not ranked"""
        return self.get_answers_by_status(requirement_id, "reviewed")

    def get_ranked_answers(self, requirement_id: str) -> List[RequirementAnswer]:
        """Get answers that have been ranked"""
        return self.get_answers_by_status(requirement_id, "ranked")

    def get_requirement_answer_stats(self, requirement_id: str) -> Dict[str, Any]:
        """
        Get statistics for answers to a specific requirement.

        Args:
            requirement_id: The requirement ID

        Returns:
            Statistics dict including counts, averages, and confirmation status
        """
        answers = self.get_answers_for_requirement(requirement_id)

        if not answers:
            return {
                "total": 0,
                "by_status": {},
                "avg_elo": 0,
                "avg_quality": 0,
                "is_confirmed": False
            }

        status_counts = {}
        for answer in answers:
            status = answer.status
            status_counts[status] = status_counts.get(status, 0) + 1

        avg_elo = sum(a.elo_rating for a in answers) / len(answers)
        avg_quality = sum(a.quality_score for a in answers) / len(answers)

        return {
            "total": len(answers),
            "by_status": status_counts,
            "avg_elo": avg_elo,
            "avg_quality": avg_quality,
            "is_confirmed": self.is_requirement_confirmed(requirement_id),
            "best_elo": max(a.elo_rating for a in answers),
            "best_quality": max(a.quality_score for a in answers)
        }

    # ========== Export Methods for Report Generation ==========

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all memory data to a serializable dictionary.
        Used by ReportGenerator to create comprehensive research reports.
        """
        return {
            "overviews": self.research_overviews,
            "meta_reviews": self.meta_reviews,
            # RequirementAnswer data
            "requirement_answers": [
                self._requirement_answer_to_dict(a)
                for a in self.requirement_answers.values()
            ],
            "confirmed_answers": {
                req_id: self._requirement_answer_to_dict(answer)
                for req_id, answer in self.confirmed_answers.items()
            },
            "confirmed_versions": {
                req_id: [self._requirement_answer_to_dict(v) for v in versions]
                for req_id, versions in self.confirmed_versions.items()
            }
        }

    def _requirement_answer_to_dict(self, answer) -> Dict[str, Any]:
        """Convert RequirementAnswer object to serializable dict"""
        if isinstance(answer, dict):
            return answer

        return {
            # Core identification
            "id": getattr(answer, "id", ""),
            "requirement_id": getattr(answer, "requirement_id", ""),
            "requirement_title": getattr(answer, "requirement_title", ""),

            # Answer content
            "answer": getattr(answer, "answer", ""),
            "rationale": getattr(answer, "rationale", ""),
            "deliverables": getattr(answer, "deliverables", {}),
            "confidence": getattr(answer, "confidence", 0.5),
            "evidence": getattr(answer, "evidence", []),
            "builds_on": getattr(answer, "builds_on", []),

            # Ranking support
            "elo_rating": getattr(answer, "elo_rating", 1200.0),
            "wins": getattr(answer, "wins", 0),
            "losses": getattr(answer, "losses", 0),

            # Status (Sequential Confirmation)
            "status": getattr(answer, "status", "generated"),

            # Review results
            "review": getattr(answer, "review", None),
            "quality_score": getattr(answer, "quality_score", 0.0),
            "novelty_score": getattr(answer, "novelty_score", 0.0),

            # Evolution tracking
            "parent_ids": getattr(answer, "parent_ids", []),
            "evolution_method": getattr(answer, "evolution_method", None),
            "iteration": getattr(answer, "iteration", 1),

            # Metadata
            "generated_at": str(getattr(answer, "generated_at", "")),
            "generation_method": getattr(answer, "generation_method", "data_based"),
            "data_sources": getattr(answer, "data_sources", []),
            "metadata": getattr(answer, "metadata", {})
        }

    def _export_ra_config(self, answer: RequirementAnswer) -> None:
        """
        Export RequirementAnswer config to RAs folder.

        Creates a JSON file with detailed information about:
        - Answer metadata
        - Tool usage (collection and analysis tools)
        - Data sources
        - Generation method and parameters

        File format: {answer_id}_config.json
        """
        if not self.ras_dir:
            return

        try:
            # Build config structure
            metadata = getattr(answer, "metadata", {}) or {}

            # Extract entity-based information (NEW!)
            entity_analysis = metadata.get("entity_analysis", {})
            data_collection = metadata.get("data_collection", {})
            data_analysis = metadata.get("data_analysis", {})

            config = {
                # Basic info
                "answer_id": answer.id,
                "requirement_id": answer.requirement_id,
                "requirement_title": answer.requirement_title,
                "generated_at": str(answer.generated_at),
                "generation_method": answer.generation_method,
                "status": answer.status,

                # === ENTITY ANALYSIS (NEW!) ===
                "entity_analysis": {
                    "primary_entities": entity_analysis.get("primary_entities", []),
                    "data_requirements": entity_analysis.get("data_requirements", []),
                    "analysis_needs": entity_analysis.get("analysis_needs", []),
                    "context_refinements": entity_analysis.get("context_refinements", {})
                },

                # === DATA COLLECTION (UPDATED!) ===
                "data_collection": {
                    "servers_used": data_collection.get("servers_used", answer.data_sources),
                    "tools_used": data_collection.get("tools", []),
                    "sources_detail": data_collection.get("sources_detail", {})
                },

                # === DATA ANALYSIS (UPDATED!) ===
                "data_analysis": {
                    "analyses_performed": data_analysis.get("analyses_performed", []),
                    "tools_used": data_analysis.get("tools", []),
                    "results": data_analysis.get("results", {})
                },

                # === GENERATION PARAMETERS ===
                "generation_params": {
                    "approach": metadata.get("approach", ""),
                    "strategy": metadata.get("strategy", ""),
                    "innovation_level": metadata.get("innovation_level", "")
                },

                # === QUALITY METRICS ===
                "quality_metrics": {
                    "elo_rating": answer.elo_rating,
                    "quality_score": answer.quality_score,
                    "novelty_score": answer.novelty_score,
                    "confidence": answer.confidence,
                    "wins": answer.wins,
                    "losses": answer.losses
                },

                # === EVOLUTION TRACKING ===
                "evolution": {
                    "parent_ids": answer.parent_ids,
                    "evolution_method": answer.evolution_method,
                    "iteration": answer.iteration
                },

                # === DEPENDENCIES & EVIDENCE ===
                "builds_on": answer.builds_on,
                "evidence": answer.evidence
            }

            # Save to file
            config_file = os.path.join(self.ras_dir, f"{answer.id}_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)

            logger.debug(f"Exported RA config: {config_file}")

        except Exception as e:
            logger.warning(f"Failed to export RA config for {answer.id}: {e}")
