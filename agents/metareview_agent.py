"""
Metareview Agent
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import Hypothesis, Review, ResearchGoal, HypothesisStatus, TournamentMatch
from ..clients import LLMClient, WebSearchClient, EmbeddingClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MetaReviewAgent(BaseAgent):
    """
    Synthesizes insights from all reviews for system improvement.
    
    Responsibilities:
    - Analyze patterns in reviews and debates
    - Provide feedback to Reflection agent
    - Improve Generation agent outputs
    - Generate research overviews
    - Identify potential research contacts
    """
    
    def __init__(self, memory: ContextMemory, config: Dict[str, Any], llm_client: Optional[LLMClient] = None, web_search: Optional[WebSearchClient] = None):
        super().__init__("MetaReviewAgent", memory, config, llm_client, web_search)
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate meta-review and research overview"""
        task_type = task.get("type", "meta_review")
        
        if task_type == "meta_review":
            return await self.generate_meta_review()
        elif task_type == "research_overview":
            return await self.generate_research_overview(task.get("research_goal"))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def generate_meta_review(self) -> Dict[str, Any]:
        """Synthesize patterns from all reviews and debates"""
        self.log("Generating meta-review")
        
        # Analyze all reviews
        all_reviews = list(self.memory.reviews.values())
        all_matches = list(self.memory.tournament_matches.values())
        
        # Identify common patterns
        patterns = self.identify_review_patterns(all_reviews)
        debate_insights = self.analyze_debates(all_matches)
        
        # Generate feedback for agents
        reflection_feedback = self.generate_reflection_feedback(patterns)
        generation_feedback = self.generate_generation_feedback(patterns)
        
        meta_review = {
            "timestamp": datetime.now(),
            "reviews_analyzed": len(all_reviews),
            "matches_analyzed": len(all_matches),
            "patterns": patterns,
            "debate_insights": debate_insights,
            "reflection_feedback": reflection_feedback,
            "generation_feedback": generation_feedback
        }
        
        self.memory.store_meta_review(meta_review)
        
        return {
            "status": "success",
            "meta_review": meta_review
        }
    
    async def generate_research_overview(
        self,
        research_goal: ResearchGoal
    ) -> Dict[str, Any]:
        """Generate comprehensive research overview"""
        self.log("Generating research overview")
        
        # Get top hypotheses
        top_hypotheses = self.memory.get_top_hypotheses(n=20)
        
        # Synthesize into research areas
        research_areas = self.synthesize_research_areas(top_hypotheses, research_goal)
        
        # Identify potential collaborators
        contacts = self.identify_research_contacts(top_hypotheses)
        
        overview = {
            "timestamp": datetime.now(),
            "research_goal": research_goal.description,
            "research_areas": research_areas,
            "top_hypotheses": [
                {
                    "id": h.id,
                    "summary": h.summary,
                    "elo_rating": h.elo_rating
                }
                for h in top_hypotheses[:10]
            ],
            "potential_contacts": contacts,
            "future_directions": self.suggest_future_directions(top_hypotheses)
        }
        
        self.memory.store_research_overview(overview)
        
        return {
            "status": "success",
            "overview": overview
        }
    
    def identify_review_patterns(self, reviews: List[Review]) -> Dict[str, Any]:
        """Identify common patterns in reviews"""
        self.log("Identifying review patterns")
        
        if not reviews:
            return {"common_issues": [], "recurring_strengths": [], "patterns": []}
        
        try:
            # Count issue frequencies
            issue_counts = {}
            strength_counts = {}
            
            for review in reviews:
                for weakness in review.weaknesses:
                    issue_counts[weakness] = issue_counts.get(weakness, 0) + 1
                
                for strength in review.strengths:
                    strength_counts[strength] = strength_counts.get(strength, 0) + 1
            
            # Find most common issues (appearing in >20% of reviews)
            threshold = len(reviews) * 0.2
            common_issues = [
                {"issue": issue, "frequency": count, "percentage": count/len(reviews)}
                for issue, count in issue_counts.items()
                if count >= threshold
            ]
            common_issues.sort(key=lambda x: x["frequency"], reverse=True)
            
            # Find recurring strengths
            recurring_strengths = [
                {"strength": strength, "frequency": count, "percentage": count/len(reviews)}
                for strength, count in strength_counts.items()
                if count >= threshold
            ]
            recurring_strengths.sort(key=lambda x: x["frequency"], reverse=True)
            
            # Analyze pass/fail rates
            pass_count = sum(1 for r in reviews if r.pass_review)
            fail_count = len(reviews) - pass_count
            
            # Identify patterns by review type
            by_type = {}
            for review in reviews:
                rtype = review.review_type
                if rtype not in by_type:
                    by_type[rtype] = {"count": 0, "passed": 0, "avg_score": []}
                
                by_type[rtype]["count"] += 1
                if review.pass_review:
                    by_type[rtype]["passed"] += 1
                
                if "score" in review.correctness_assessment:
                    by_type[rtype]["avg_score"].append(review.correctness_assessment["score"])
            
            # Calculate averages
            for rtype in by_type:
                scores = by_type[rtype]["avg_score"]
                by_type[rtype]["avg_correctness"] = sum(scores) / len(scores) if scores else 0
                by_type[rtype]["pass_rate"] = by_type[rtype]["passed"] / by_type[rtype]["count"]
                del by_type[rtype]["avg_score"]  # Remove raw scores
            
            patterns = {
                "common_issues": common_issues[:10],
                "recurring_strengths": recurring_strengths[:10],
                "total_reviews": len(reviews),
                "pass_rate": pass_count / len(reviews),
                "fail_rate": fail_count / len(reviews),
                "by_review_type": by_type,
                "top_3_issues": [issue["issue"] for issue in common_issues[:3]]
            }
            
            self.log(f"Identified {len(common_issues)} common issues, {len(recurring_strengths)} strengths")
            return patterns
            
        except Exception as e:
            self.log(f"Error identifying patterns: {e}", "error")
            return {"common_issues": [], "recurring_strengths": [], "patterns": []}
    
    def analyze_debates(self, matches: List[TournamentMatch]) -> Dict[str, Any]:
        """Analyze patterns in tournament debates"""
        # TODO: Implement debate analysis
        return {"key_decision_factors": []}
    
    def generate_reflection_feedback(self, patterns: Dict) -> Dict[str, Any]:
        """Generate feedback for Reflection agent"""
        # TODO: Implement feedback generation
        return {"focus_areas": [], "critical_checks": []}
    
    def generate_generation_feedback(self, patterns: Dict) -> Dict[str, Any]:
        """Generate feedback for Generation agent"""
        # TODO: Implement feedback generation
        return {"avoid_patterns": [], "explore_areas": []}
    
    def synthesize_research_areas(
        self,
        hypotheses: List[Hypothesis],
        research_goal: ResearchGoal
    ) -> List[Dict]:
        """Synthesize top hypotheses into research areas"""
        self.log("Synthesizing research areas from top hypotheses")
        
        if not hypotheses:
            return []
        
        try:
            # Group hypotheses by category
            by_category = {}
            for hyp in hypotheses:
                category = hyp.category or "Uncategorized"
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(hyp)
            
            # Create research areas
            research_areas = []
            
            for category, hyps in by_category.items():
                if len(hyps) < 2:  # Skip small groups
                    continue
                
                # Sort by Elo rating
                sorted_hyps = sorted(hyps, key=lambda h: h.elo_rating, reverse=True)
                top_hyps = sorted_hyps[:5]
                
                # Calculate average metrics
                avg_elo = sum(h.elo_rating for h in top_hyps) / len(top_hyps)
                total_matches = sum(h.wins + h.losses for h in top_hyps)
                
                # Get common themes from summaries
                summaries = [h.summary for h in top_hyps]
                
                area = {
                    "area_name": category,
                    "hypothesis_count": len(hyps),
                    "top_hypotheses": [
                        {
                            "id": h.id,
                            "summary": h.summary,
                            "elo_rating": h.elo_rating,
                            "wins": h.wins
                        }
                        for h in top_hyps
                    ],
                    "average_elo": avg_elo,
                    "total_evaluations": total_matches,
                    "promising": avg_elo > 1300,
                    "key_themes": summaries[:3],
                    "research_priority": "high" if avg_elo > 1400 else "medium" if avg_elo > 1200 else "low"
                }
                
                research_areas.append(area)
            
            # Sort by average Elo
            research_areas.sort(key=lambda x: x["average_elo"], reverse=True)
            
            self.log(f"Synthesized {len(research_areas)} research areas")
            return research_areas
            
        except Exception as e:
            self.log(f"Error synthesizing research areas: {e}", "error")
            return []
    
    def identify_research_contacts(self, hypotheses: List[Hypothesis]) -> List[Dict]:
        """Identify potential research collaborators"""
        self.log("Identifying potential research collaborators")
        
        if not hypotheses:
            return []
        
        try:
            contacts = []
            author_expertise = {}
            
            # Extract cited authors from hypothesis metadata
            for hyp in hypotheses:
                citations = hyp.metadata.get("key_citations", [])
                papers_used = hyp.metadata.get("papers_used", 0)
                
                for citation in citations:
                    # Parse author from citation (simple extraction)
                    # Format: "Author et al., PMID/arXiv ID"
                    if " et al." in citation:
                        author = citation.split(" et al.")[0].strip()
                        
                        if author not in author_expertise:
                            author_expertise[author] = {
                                "name": author,
                                "citations": 0,
                                "relevant_hypotheses": [],
                                "research_areas": set()
                            }
                        
                        author_expertise[author]["citations"] += 1
                        author_expertise[author]["relevant_hypotheses"].append(hyp.id)
                        author_expertise[author]["research_areas"].add(hyp.category)
            
            # Convert to list and sort by relevance
            for author, info in author_expertise.items():
                if info["citations"] >= 2:  # At least 2 citations
                    contacts.append({
                        "name": info["name"],
                        "citation_count": info["citations"],
                        "relevant_hypothesis_count": len(info["relevant_hypotheses"]),
                        "research_areas": list(info["research_areas"]),
                        "relevance_score": info["citations"] * len(info["research_areas"]),
                        "suggested_collaboration": f"Expert in {', '.join(list(info['research_areas'])[:2])}"
                    })
            
            # Sort by relevance
            contacts.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            self.log(f"Identified {len(contacts)} potential collaborators")
            return contacts[:20]  # Top 20
            
        except Exception as e:
            self.log(f"Error identifying research contacts: {e}", "error")
            return []
    
    def suggest_future_directions(self, hypotheses: List[Hypothesis]) -> List[str]:
        """Suggest future research directions"""
        self.log("Suggesting future research directions")
        
        if not hypotheses:
            return []
        
        try:
            suggestions = []
            
            # Sort by Elo rating
            top_hypotheses = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:10]
            
            # Group by category
            by_category = {}
            for hyp in top_hypotheses:
                category = hyp.category or "General"
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(hyp)
            
            # Suggest directions for each strong category
            for category, hyps in by_category.items():
                if len(hyps) >= 2:
                    suggestions.append(
                        f"Further explore {category}: {len(hyps)} promising hypotheses identified"
                    )
                    
                    # Find common themes
                    summaries = ' '.join([h.summary for h in hyps[:3]])
                    
                    # Check for specific keywords
                    if "CRISPR" in summaries or "gene edit" in summaries:
                        suggestions.append(
                            f"Investigate delivery mechanisms for {category} approaches"
                        )
                    
                    if "drug" in summaries.lower() or "compound" in summaries.lower():
                        suggestions.append(
                            f"Screen compound libraries for {category} targets"
                        )
                    
                    if "machine learning" in summaries.lower() or "AI" in summaries:
                        suggestions.append(
                            f"Apply deep learning to {category} prediction tasks"
                        )
            
            # Suggest combinations
            if len(by_category) >= 2:
                categories = list(by_category.keys())[:3]
                suggestions.append(
                    f"Explore combinations: {' + '.join(categories)}"
                )
            
            # Check for gaps - what's not being explored
            all_categories = set([h.category for h in hypotheses if h.category])
            common_biotech_areas = {
                "Gene Therapy", "Drug Discovery", "Immunotherapy", 
                "Diagnostics", "Biomarkers", "Synthetic Biology"
            }
            
            unexplored = common_biotech_areas - all_categories
            if unexplored:
                suggestions.append(
                    f"Consider unexplored areas: {', '.join(list(unexplored)[:3])}"
                )
            
            # Suggest validation experiments
            high_elo_hyps = [h for h in hypotheses if h.elo_rating > 1400]
            if high_elo_hyps:
                suggestions.append(
                    f"Design validation experiments for {len(high_elo_hyps)} top-rated hypotheses"
                )
            
            # Suggest interdisciplinary approaches
            if len(all_categories) <= 3:
                suggestions.append(
                    "Explore interdisciplinary approaches combining biology, chemistry, and computation"
                )
            
            self.log(f"Generated {len(suggestions)} future direction suggestions")
            return suggestions
            
        except Exception as e:
            self.log(f"Error suggesting future directions: {e}", "error")
            return []

