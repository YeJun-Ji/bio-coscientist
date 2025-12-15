"""
Proximity Agent
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import Hypothesis, Review, ResearchGoal, HypothesisStatus, TournamentMatch
from ..clients import LLMClient, WebSearchClient, EmbeddingClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ProximityAgent(BaseAgent):
    """
    Computes similarity between hypotheses and builds proximity graph.
    
    Features:
    - Calculate semantic similarity between hypotheses using embeddings
    - Build proximity graph for clustering
    - Assist in organizing diverse tournament matches
    - Help identify related concepts
    """
    
    def __init__(
        self,
        memory: ContextMemory,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        web_search: Optional[WebSearchClient] = None,
        embedding_client: Optional[EmbeddingClient] = None
    ):
        super().__init__("ProximityAgent", memory, config, llm_client, web_search)
        self.embedding_client = embedding_client
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Compute proximity graph for hypotheses"""
        hypotheses = task.get("hypotheses", [])
        research_goal = task.get("research_goal")
        
        self.log(f"Computing proximity graph for {len(hypotheses)} hypotheses")
        
        # Compute pairwise similarities
        proximity_graph = await self.compute_proximity_graph(hypotheses, research_goal)
        
        # Identify clusters
        clusters = self.identify_clusters(proximity_graph)
        
        return {
            "status": "success",
            "proximity_graph": proximity_graph,
            "clusters": clusters
        }
    
    async def compute_proximity_graph(
        self,
        hypotheses: List[Hypothesis],
        research_goal: ResearchGoal
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise similarity scores using embeddings.
        
        Returns a graph where graph[hyp_a_id][hyp_b_id] = similarity_score
        """
        self.log(f"Computing pairwise similarities for {len(hypotheses)} hypotheses")
        
        if not self.embedding_client:
            self.log("Warning: No embedding client available, using fallback LLM comparison")
            return await self._compute_graph_with_llm(hypotheses, research_goal)
        
        # Get all hypothesis texts for batch embedding
        hypothesis_texts = [self._format_hypothesis_for_embedding(h) for h in hypotheses]
        
        # Get embeddings in batch
        try:
            embeddings = await self.embedding_client.get_embeddings(hypothesis_texts)
            self.log(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            self.log(f"Embedding generation failed: {e}, falling back to LLM")
            return await self._compute_graph_with_llm(hypotheses, research_goal)
        
        # Compute pairwise similarities
        graph = {}
        for i, hyp_a in enumerate(hypotheses):
            graph[hyp_a.id] = {}
            for j, hyp_b in enumerate(hypotheses):
                if i != j:  # Skip self-similarity
                    similarity = self.embedding_client.cosine_similarity(
                        embeddings[i], embeddings[j]
                    )
                    graph[hyp_a.id][hyp_b.id] = similarity
        
        return graph
    
    def _format_hypothesis_for_embedding(self, hypothesis: Hypothesis) -> str:
        """
        Format hypothesis for embedding generation.
        Combines summary and key details for better semantic representation.
        """
        parts = [hypothesis.summary]
        if hypothesis.content:
            parts.append(hypothesis.content[:500])  # Limit content length
        if hypothesis.category:
            parts.append(f"Category: {hypothesis.category}")
        return " | ".join(parts)
    
    async def _compute_graph_with_llm(
        self,
        hypotheses: List[Hypothesis],
        research_goal: ResearchGoal
    ) -> Dict[str, Dict[str, float]]:
        """
        Fallback: Compute similarity using LLM when embeddings unavailable.
        This is slower but doesn't require embedding model.
        """
        graph = {}
        for i, hyp_a in enumerate(hypotheses):
            graph[hyp_a.id] = {}
            for j, hyp_b in enumerate(hypotheses):
                if i != j:
                    similarity = await self.compute_similarity(hyp_a, hyp_b, research_goal)
                    graph[hyp_a.id][hyp_b.id] = similarity
        return graph
    
    async def compute_similarity(
        self,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        research_goal: ResearchGoal
    ) -> float:
        """
        Compute similarity between two hypotheses using LLM.
        Used as fallback when embeddings are unavailable.
        """
        if not self.llm:
            return 0.5  # No comparison possible
        
        prompt = f"""Compare these two research hypotheses and rate their similarity from 0.0 (completely different) to 1.0 (nearly identical).

Research Goal: {research_goal.description}

Hypothesis A: {hyp_a.summary}
{hyp_a.content[:300] if hyp_a.content else ''}

Hypothesis B: {hyp_b.summary}
{hyp_b.content[:300] if hyp_b.content else ''}

Consider:
- Shared concepts and methodologies
- Similar research questions
- Overlapping target outcomes

Respond with only a number between 0.0 and 1.0."""
        
        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10,
                purpose="similarity calculation"
            )
            similarity = float(response.strip())
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except:
            return 0.5  # Default similarity on error
    
    def identify_clusters(self, proximity_graph: Dict, threshold: float = 0.7) -> List[List[str]]:
        """Identify clusters of similar hypotheses using threshold-based clustering"""
        self.log("Identifying hypothesis clusters")
        
        if not proximity_graph:
            return []
        
        try:
            clusters = []
            visited = set()
            
            # Get all hypothesis IDs
            all_hyp_ids = set(proximity_graph.keys())
            
            # Threshold-based clustering
            for hyp_id in all_hyp_ids:
                if hyp_id in visited:
                    continue
                
                # Start a new cluster
                cluster = [hyp_id]
                visited.add(hyp_id)
                
                # Find all similar hypotheses
                for other_id, similarity in proximity_graph.get(hyp_id, {}).items():
                    if other_id not in visited and similarity >= threshold:
                        cluster.append(other_id)
                        visited.add(other_id)
                
                # Only keep clusters with multiple members
                if len(cluster) >= 2:
                    clusters.append(cluster)
            
            self.log(f"Identified {len(clusters)} clusters with threshold={threshold}")
            return clusters
            
        except Exception as e:
            self.log(f"Error in clustering: {e}", "error")
            return []
