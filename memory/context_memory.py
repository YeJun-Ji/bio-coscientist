"""
Context Memory System - Maintains research context and history
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from ..core import Hypothesis, Review, ResearchGoal, TournamentMatch

logger = logging.getLogger(__name__)


class ContextMemory:
    """
    Maintains research context and historical information.
    Stores hypotheses, reviews, tournament results, and meta-insights.
    """
    
    def __init__(self, storage_path: str = "./research_memory"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.reviews: Dict[str, Review] = {}
        self.tournament_matches: Dict[str, TournamentMatch] = {}
        self.meta_reviews: List[Dict] = []
        self.research_overviews: List[Dict] = []
        
        logger.info(f"ContextMemory initialized at {storage_path}")
    
    def store_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Store a hypothesis in memory"""
        self.hypotheses[hypothesis.id] = hypothesis
        logger.info(f"Stored hypothesis {hypothesis.id}")
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve a hypothesis by ID"""
        return self.hypotheses.get(hypothesis_id)
    
    def get_top_hypotheses(self, n: int = 10) -> List[Hypothesis]:
        """Get top N hypotheses by Elo rating"""
        sorted_hypotheses = sorted(
            self.hypotheses.values(),
            key=lambda h: h.elo_rating,
            reverse=True
        )
        return sorted_hypotheses[:n]
    
    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Get all active hypotheses (not rejected or archived)"""
        from ..core import HypothesisStatus
        return [
            h for h in self.hypotheses.values()
            if h.status not in [HypothesisStatus.REJECTED, HypothesisStatus.ARCHIVED, HypothesisStatus.FAILED_REVIEW]
        ]
    
    def store_review(self, review: Review) -> None:
        """Store a review in memory"""
        self.reviews[review.review_id] = review
        logger.info(f"Stored review {review.review_id} for hypothesis {review.hypothesis_id}")
    
    def get_reviews_for_hypothesis(self, hypothesis_id: str) -> List[Review]:
        """Get all reviews for a specific hypothesis"""
        return [r for r in self.reviews.values() if r.hypothesis_id == hypothesis_id]
    
    def store_tournament_match(self, match: TournamentMatch) -> None:
        """Store a tournament match"""
        self.tournament_matches[match.match_id] = match
        logger.info(f"Stored tournament match {match.match_id}")
    
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
    
    def save_to_disk(self) -> None:
        """Persist memory to disk"""
        # Implementation for saving to JSON/database
        pass
    
    def load_from_disk(self) -> None:
        """Load memory from disk"""
        # Implementation for loading from JSON/database
        pass
