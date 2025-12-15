"""
Evolution Agent
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import Hypothesis, Review, ResearchGoal, HypothesisStatus, TournamentMatch
from ..clients import LLMClient, WebSearchClient, EmbeddingClient

from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """
    Continuously refines and improves hypotheses.
    
    Refinement strategies:
    - Enhancement through grounding with literature
    - Coherence, practicality, and feasibility improvements
    - Inspiration from existing hypotheses
    - Combination of multiple hypotheses
    - Simplification for easier testing
    - Out-of-box thinking
    """
    
    def __init__(self, memory: ContextMemory, config: Dict[str, Any], llm_client: Optional[LLMClient] = None, web_search: Optional[WebSearchClient] = None):
        super().__init__("EvolutionAgent", memory, config, llm_client, web_search)
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve and improve hypotheses"""
        top_hypotheses = task.get("top_hypotheses", [])
        evolution_method = task.get("method", "all")
        
        self.log(f"Evolving {len(top_hypotheses)} hypotheses using method: {evolution_method}")
        
        evolved_hypotheses = []
        
        # Apply various evolution strategies
        if evolution_method in ["all", "grounding"]:
            evolved = await self.enhance_through_grounding(top_hypotheses)
            evolved_hypotheses.extend(evolved)
        
        if evolution_method in ["all", "coherence"]:
            evolved = await self.improve_coherence(top_hypotheses)
            evolved_hypotheses.extend(evolved)
        
        if evolution_method in ["all", "combination"]:
            evolved = await self.combine_hypotheses(top_hypotheses)
            evolved_hypotheses.extend(evolved)
        
        if evolution_method in ["all", "simplification"]:
            evolved = await self.simplify_hypotheses(top_hypotheses)
            evolved_hypotheses.extend(evolved)
        
        if evolution_method in ["all", "divergent"]:
            evolved = await self.generate_divergent(top_hypotheses)
            evolved_hypotheses.extend(evolved)
        
        # Store evolved hypotheses
        for hyp in evolved_hypotheses:
            self.memory.store_hypothesis(hyp)
        
        return {
            "status": "success",
            "evolved_count": len(evolved_hypotheses),
            "hypotheses": evolved_hypotheses
        }
    
    async def enhance_through_grounding(
        self,
        hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Improve hypotheses by grounding with literature"""
        self.log("Enhancing hypotheses through grounding")
        
        if not self.llm or not self.web_search:
            self.log("LLM or web search not configured", "warning")
            return []
        
        try:
            enhanced_hypotheses = []
            
            for hypothesis in hypotheses[:5]:  # Top 5
                self.log(f"Grounding {hypothesis.id}")
                
                # Step 1: Identify weaknesses from reviews
                reviews = self.memory.get_reviews_for_hypothesis(hypothesis.id)
                weaknesses = []
                for review in reviews[-2:]:
                    weaknesses.extend(review.weaknesses)
                
                if not weaknesses:
                    weaknesses = ["Need more specific details", "Experimental approach unclear"]
                
                # Step 2: Search literature for each weakness
                search_query = f"{hypothesis.summary} {' '.join(weaknesses[:2])}"
                papers = await self.web_search.search_all(search_query, max_results=5)
                
                if not papers:
                    self.log(f"No papers found for {hypothesis.id}", "warning")
                    continue
                
                papers_text = self.web_search.format_papers_for_llm(papers[:3])
                
                # Step 3: Use literature to enhance hypothesis
                enhance_prompt = f"""Enhance this hypothesis using insights from recent literature.

Original Hypothesis: {hypothesis.summary}
Full Content: {hypothesis.content}

Identified Weaknesses:
{json.dumps(weaknesses, indent=2)}

Relevant Literature:
{papers_text}

Improve the hypothesis by:
1. Addressing weaknesses with specific details from literature
2. Adding experimental methods from papers
3. Including dosage/parameters when applicable
4. Maintaining the core idea while making it more concrete

Provide as JSON:
{{
  "enhanced_content": "Improved hypothesis (4-6 sentences)",
  "enhanced_summary": "One sentence summary",
  "improvements": ["improvement 1 with citation", "improvement 2"],
  "weaknesses_addressed": ["weakness addressed"],
  "key_citations": ["Author et al., PMID/arXiv ID"]
}}"""
                
                response = await self.llm.generate_json(
                    messages=[{"role": "user", "content": enhance_prompt}],
                    temperature=0.5,
                    purpose="literature grounding",
                    use_cache=True  # 문헌 컨텍스트를 캐시하여 여러 가설 처리 시 재사용
                )
                
                # Create enhanced hypothesis
                enhanced = Hypothesis(
                    id=f"hyp_ground_{datetime.now().timestamp()}_{hypothesis.id[-6:]}",
                    content=response["enhanced_content"],
                    category=hypothesis.category,
                    summary=response["enhanced_summary"],
                    generated_at=datetime.now(),
                    parent_ids=[hypothesis.id],
                    evolution_method="literature_grounding",  # Set evolution_method field
                    metadata={
                        "evolution_method": "literature_grounding",
                        "parent_id": hypothesis.id,
                        "improvements": response.get("improvements", []),
                        "weaknesses_addressed": response.get("weaknesses_addressed", []),
                        "key_citations": response.get("key_citations", []),
                        "papers_used": len(papers)
                    }
                )
                
                enhanced_hypotheses.append(enhanced)
                self.memory.store_hypothesis(enhanced)
            
            self.log(f"Enhanced {len(enhanced_hypotheses)} hypotheses with literature")
            return enhanced_hypotheses
            
        except Exception as e:
            self.log(f"Error in literature grounding: {e}", "error")
            return []
    
    async def improve_coherence(
        self,
        hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Improve coherence, practicality, and feasibility"""
        self.log("Improving hypothesis coherence and feasibility")
        
        if not self.llm:
            self.log("LLM not configured", "warning")
            return []
        
        try:
            improved_hypotheses = []
            
            for hypothesis in hypotheses[:5]:  # Top 5
                self.log(f"Improving coherence for {hypothesis.id}")
                
                # Get reviews to identify issues
                reviews = self.memory.get_reviews_for_hypothesis(hypothesis.id)
                issues = []
                for review in reviews[-2:]:
                    issues.extend(review.weaknesses)
                
                improve_prompt = f"""Improve the internal coherence and practicality of this hypothesis.

Original Hypothesis: {hypothesis.summary}
Full Content: {hypothesis.content}

Identified Issues:
{chr(10).join([f"- {i}" for i in issues]) if issues else "None identified"}

Improve by:
1. Ensuring internal consistency (no contradictions)
2. Making experimental approach more practical
3. Adding specific parameters/conditions
4. Clarifying vague statements
5. Ensuring all parts connect logically

Provide as JSON:
{{
  "improved_content": "Revised hypothesis (3-5 sentences)",
  "improved_summary": "One sentence summary",
  "coherence_improvements": ["improvement 1", "improvement 2"],
  "feasibility_improvements": ["made more practical by..."],
  "clarifications": ["clarified aspect 1"]
}}"""
                
                response = await self.llm.generate_json(
                    messages=[{"role": "user", "content": improve_prompt}],
                    temperature=0.5,
                    purpose="coherence improvement"
                )
                
                improved = Hypothesis(
                    id=f"hyp_coherent_{datetime.now().timestamp()}_{hypothesis.id[-6:]}",
                    content=response["improved_content"],
                    category=hypothesis.category,
                    summary=response["improved_summary"],
                    generated_at=datetime.now(),
                    parent_ids=[hypothesis.id],
                    evolution_method="coherence_improvement",
                    metadata={
                        "evolution_method": "coherence_improvement",
                        "parent_id": hypothesis.id,
                        "coherence_improvements": response.get("coherence_improvements", []),
                        "feasibility_improvements": response.get("feasibility_improvements", []),
                        "clarifications": response.get("clarifications", [])
                    }
                )
                
                improved_hypotheses.append(improved)
                self.memory.store_hypothesis(improved)
            
            self.log(f"Improved coherence for {len(improved_hypotheses)} hypotheses")
            return improved_hypotheses
            
        except Exception as e:
            self.log(f"Error improving coherence: {e}", "error")
            return []
    
    async def combine_hypotheses(
        self,
        hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Combine best aspects of multiple hypotheses"""
        self.log("Combining top hypotheses")
        
        if not self.llm or len(hypotheses) < 2:
            self.log("Need at least 2 hypotheses and LLM to combine", "warning")
            return []
        
        try:
            evolved_hypotheses = []
            
            # Combine in groups of 2-3
            for i in range(0, min(len(hypotheses), 6), 2):
                hyps_to_combine = hypotheses[i:i+3] if i+2 < len(hypotheses) else hypotheses[i:i+2]
                
                if len(hyps_to_combine) < 2:
                    continue
                
                # Format hypotheses for combination
                hyps_text = "\n\n".join([
                    f"""Hypothesis {j+1} (ID: {h.id}, Elo: {h.elo_rating})
Summary: {h.summary}
Content: {h.content}
Category: {h.category}"""
                    for j, h in enumerate(hyps_to_combine)
                ])
                
                # Extract strengths from reviews
                strengths_by_hyp = []
                for h in hyps_to_combine:
                    reviews = self.memory.get_reviews_for_hypothesis(h.id)
                    strengths = []
                    for review in reviews[-2:]:  # Last 2 reviews
                        strengths.extend(review.strengths)
                    strengths_by_hyp.append(strengths)
                
                combine_prompt = f"""Combine the best aspects of these hypotheses into 1 improved hypothesis.

{hyps_text}

Strengths identified in reviews:
{json.dumps([{f"Hypothesis {j+1}": s} for j, s in enumerate(strengths_by_hyp)], indent=2)}

Create a new hypothesis that:
1. Integrates the strongest elements from each
2. Eliminates weaknesses and redundancies
3. Is more comprehensive and testable
4. Maintains scientific rigor

Provide as JSON:
{{
  "content": "Combined hypothesis (3-5 sentences)",
  "summary": "One sentence summary",
  "category": "Category",
  "parent_ids": ["{hyps_to_combine[0].id}", "{hyps_to_combine[1].id}"],
  "improvements": ["improvement 1", "improvement 2"],
  "retained_strengths": ["strength from original hypotheses"]
}}"""
                
                response = await self.llm.generate_json(
                    messages=[{"role": "user", "content": combine_prompt}],
                    temperature=0.6,
                    purpose="hypothesis combination"
                )
                
                # Create combined hypothesis
                combined = Hypothesis(
                    id=f"hyp_combined_{datetime.now().timestamp()}_{i}",
                    content=response["content"],
                    category=response.get("category", hyps_to_combine[0].category),
                    summary=response["summary"],
                    generated_at=datetime.now(),
                    parent_ids=[h.id for h in hyps_to_combine],
                    evolution_method="combination",
                    metadata={
                        "evolution_method": "combination",
                        "improvements": response.get("improvements", []),
                        "retained_strengths": response.get("retained_strengths", [])
                    }
                )
                
                evolved_hypotheses.append(combined)
                self.memory.store_hypothesis(combined)
            
            self.log(f"Created {len(evolved_hypotheses)} combined hypotheses")
            return evolved_hypotheses
            
        except Exception as e:
            self.log(f"Error in hypothesis combination: {e}", "error")
            return []
    
    async def simplify_hypotheses(
        self,
        hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Simplify hypotheses for easier verification"""
        self.log("Simplifying hypotheses")
        
        if not self.llm:
            self.log("LLM not configured", "warning")
            return []
        
        try:
            simplified_hypotheses = []
            
            for hypothesis in hypotheses[:5]:
                self.log(f"Simplifying {hypothesis.id}")
                
                simplify_prompt = f"""Simplify this hypothesis to make it more testable and verifiable.

Original Hypothesis: {hypothesis.summary}
Full Content: {hypothesis.content}

Simplify by:
1. Reducing complexity while keeping core idea
2. Focusing on ONE main testable claim
3. Removing unnecessary details or assumptions
4. Making it more concrete and specific
5. Ensuring it can be tested with available methods

Provide as JSON:
{{
  "simplified_content": "Simplified hypothesis (2-3 sentences)",
  "simplified_summary": "One sentence summary",
  "core_claim": "The single most important testable claim",
  "removed_complexity": ["what was removed/simplified"],
  "test_method": "Suggested simple test approach"
}}"""
                
                response = await self.llm.generate_json(
                    messages=[{"role": "user", "content": simplify_prompt}],
                    temperature=0.4,
                    purpose="hypothesis simplification"
                )
                
                simplified = Hypothesis(
                    id=f"hyp_simple_{datetime.now().timestamp()}_{hypothesis.id[-6:]}",
                    content=response["simplified_content"],
                    category=hypothesis.category,
                    summary=response["simplified_summary"],
                    generated_at=datetime.now(),
                    parent_ids=[hypothesis.id],
                    evolution_method="simplification",
                    metadata={
                        "evolution_method": "simplification",
                        "parent_id": hypothesis.id,
                        "core_claim": response.get("core_claim", ""),
                        "removed_complexity": response.get("removed_complexity", []),
                        "test_method": response.get("test_method", "")
                    }
                )
                
                simplified_hypotheses.append(simplified)
                self.memory.store_hypothesis(simplified)
            
            self.log(f"Simplified {len(simplified_hypotheses)} hypotheses")
            return simplified_hypotheses
            
        except Exception as e:
            self.log(f"Error simplifying hypotheses: {e}", "error")
            return []
    
    async def generate_divergent(
        self,
        hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Generate divergent, out-of-box ideas"""
        self.log("Generating divergent hypotheses")
        
        if not self.llm or not hypotheses:
            self.log("Need LLM and existing hypotheses", "warning")
            return []
        
        try:
            # Analyze existing approaches
            existing_approaches = []
            existing_categories = set()
            
            for hyp in hypotheses[:10]:
                existing_categories.add(hyp.category)
                # Extract methodology from metadata or content
                if "methodology" in hyp.metadata:
                    existing_approaches.append(hyp.metadata["methodology"])
            
            divergent_prompt = f"""Generate radically different research approaches.

Existing Hypotheses Sample:
{chr(10).join([f"- {h.summary}" for h in hypotheses[:7]])}

Existing Categories: {', '.join(existing_categories)}

Generate 3-4 divergent hypotheses that:
1. Use COMPLETELY DIFFERENT methodologies
2. Challenge fundamental assumptions of existing hypotheses
3. Explore unconventional or interdisciplinary approaches
4. Think "outside the box" but remain scientifically plausible

Examples of divergence:
- If existing use drugs, propose physical/mechanical approaches
- If existing focus on genes, propose epigenetic/environmental factors
- If existing use in vivo, propose computational/in silico methods
- Combine fields that haven't been combined before

Provide as JSON:
{{
  "divergent_hypotheses": [
    {{
      "content": "Divergent hypothesis (3-4 sentences)",
      "summary": "One sentence summary",
      "category": "New or unconventional category",
      "divergence_type": "methodology/assumption/interdisciplinary",
      "why_divergent": "How this differs from existing approaches"
    }}
  ]
}}"""
            
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": divergent_prompt}],
                temperature=0.9,  # High temperature for creativity
                purpose="divergent hypothesis generation"
            )
            
            divergent_hypotheses = []
            for i, hyp_data in enumerate(response.get("divergent_hypotheses", [])):
                divergent = Hypothesis(
                    id=f"hyp_divergent_{datetime.now().timestamp()}_{i}",
                    content=hyp_data["content"],
                    category=hyp_data.get("category", "divergent"),
                    summary=hyp_data["summary"],
                    generated_at=datetime.now(),
                    parent_ids=[hypotheses[0].id] if hypotheses else [],
                    evolution_method="divergent_thinking",
                    metadata={
                        "evolution_method": "divergent_thinking",
                        "divergence_type": hyp_data.get("divergence_type", ""),
                        "why_divergent": hyp_data.get("why_divergent", "")
                    }
                )
                
                divergent_hypotheses.append(divergent)
                self.memory.store_hypothesis(divergent)
            
            self.log(f"Generated {len(divergent_hypotheses)} divergent hypotheses")
            return divergent_hypotheses
            
        except Exception as e:
            self.log(f"Error generating divergent hypotheses: {e}", "error")
            return []
