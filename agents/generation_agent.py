"""
Generation Agent - Creates novel hypotheses
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core import Hypothesis, Review, ResearchGoal, HypothesisStatus
from ..clients import LLMClient, WebSearchClient, EmbeddingClient
from ..memory import ContextMemory
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)




class GenerationAgent(BaseAgent):
    """
    Generates novel hypotheses and research proposals.
    
    Techniques:
    - Literature exploration via web search
    - Simulated scientific debates
    - Iterative assumptions identification
    - Research expansion based on existing hypotheses
    """
    
    def __init__(self, memory: ContextMemory, config: Dict[str, Any], llm_client: Optional[LLMClient] = None, web_search: Optional[WebSearchClient] = None):
        super().__init__("GenerationAgent", memory, config, llm_client, web_search)
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hypotheses based on research goal"""
        research_goal = task.get("research_goal")
        existing_hypotheses = task.get("existing_hypotheses", [])
        meta_review_feedback = task.get("meta_review_feedback")
        
        self.log(f"Generating hypotheses for goal: {research_goal.description}")
        
        # Generate hypotheses using various methods
        hypotheses = []
        
        # 1. Literature-based generation
        lit_hypotheses = await self.generate_from_literature(research_goal)
        hypotheses.extend(lit_hypotheses)
        
        # 2. Debate-based generation
        debate_hypotheses = await self.generate_from_debates(research_goal)
        hypotheses.extend(debate_hypotheses)
        
        # 3. Assumptions-based generation
        assumption_hypotheses = await self.generate_from_assumptions(research_goal)
        hypotheses.extend(assumption_hypotheses)
        
        # 4. Expansion from existing hypotheses
        if existing_hypotheses:
            expansion_hypotheses = await self.expand_research_space(
                research_goal, existing_hypotheses, meta_review_feedback
            )
            hypotheses.extend(expansion_hypotheses)
        
        # Store generated hypotheses in memory
        for hyp in hypotheses:
            self.memory.store_hypothesis(hyp)
        
        return {
            "status": "success",
            "hypotheses_generated": len(hypotheses),
            "hypotheses": hypotheses
        }
    
    async def generate_from_literature(self, research_goal: ResearchGoal) -> List[Hypothesis]:
        """문헌 기반 가설 생성 with LLM Tool Calling"""
        self.log("⚙️ Literature exploration with tool calling...")
        
        if not self.llm:
            self.log("LLM not configured, skipping literature-based generation", "warning")
            return []
        
        if not self.web_search:
            self.log("WebSearch not configured, skipping literature-based generation", "warning")
            return []
        
        # Ensure WebSearchClient context is properly managed
        async with self.web_search:
            try:
                return await self._generate_from_literature_impl(research_goal)
            except Exception as e:
                self.log(f"Error generating from literature: {e}", "error")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}", "debug")
                return []
    
    async def _generate_from_literature_impl(self, research_goal: ResearchGoal) -> List[Hypothesis]:
        """Internal implementation of literature-based generation"""
        try:
            # Define search tools for LLM
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_pubmed",
                        "description": "Search PubMed for recent, highly-cited scientific papers in medical and life sciences. Prioritizes papers from the last 5 years with high citation counts.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query with scientific keywords (e.g., 'CRISPR gene editing cancer')"
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of papers to return",
                                    "default": 10
                                },
                                "year_range": {
                                    "type": "integer",
                                    "description": "Number of years back to search (default: 5 for recent papers)",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_semantic_scholar",
                        "description": "Search Semantic Scholar for highly-cited academic papers. Returns papers sorted by citation count, prioritizing recent influential work from the last 5 years.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of papers",
                                    "default": 10
                                },
                                "year_range": {
                                    "type": "integer",
                                    "description": "Number of years back to search (default: 5 for recent papers)",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            
            # Prepare focus areas
            focus_areas_str = ""
            if hasattr(research_goal, 'focus_areas') and research_goal.focus_areas:
                if isinstance(research_goal.focus_areas, list):
                    focus_areas_str = ", ".join(str(x) for x in research_goal.focus_areas)
                else:
                    focus_areas_str = str(research_goal.focus_areas)
            
            # Step 1: Ask LLM to search literature and analyze it
            initial_message = f"""You are a scientific research assistant analyzing literature to generate novel hypotheses.

Research Goal: {research_goal.description}
Domain: {research_goal.domain}
Focus Areas: {focus_areas_str}

Your task:
1. Use the search tools to find relevant scientific papers from the LAST 5 YEARS
2. Prioritize highly-cited papers that have made significant impact
3. Analyze the papers to identify research gaps and opportunities
4. Generate 3 novel hypotheses based on the literature

Search Strategy:
- Use year_range=5 to get recent papers (2020-2025)
- The tools automatically prioritize papers with high citation counts
- Search for multiple aspects of the research goal to ensure comprehensive coverage

Start by searching for papers related to this research goal. Choose appropriate search queries and tools."""
            
            self.log("Requesting LLM to search literature...")
            
            # Step 2: Get LLM response with tool calls
            response = await self.llm.generate_with_tools(
                messages=[{"role": "user", "content": initial_message}],
                tools=tools,
                temperature=0.7,
                max_iterations=3,
                purpose="literature-based hypothesis generation"
            )
            
            # Step 3: Execute tool calls if requested
            tool_results = []
            if response.get('pending_tool_calls'):
                self.log(f"Executing {len(response['tool_calls'])} literature searches...")
                
                # WebSearchClient is already in context from outer function
                for tool_call in response['tool_calls']:
                    tool_name = tool_call['name']
                    args = tool_call['arguments']
                    
                    try:
                        if tool_name == 'search_pubmed':
                            papers = await self.web_search.search_pubmed(
                                query=args.get('query', ''),
                                max_results=args.get('max_results', 10),
                                year_range=args.get('year_range', 5)
                            )
                            result = self.web_search.format_papers_for_context(papers, max_papers=5)
                            tool_results.append({
                                'tool_call_id': tool_call['id'],
                                'output': result
                            })
                            self.log(f"✓ PubMed search: found {len(papers)} papers")
                            
                        elif tool_name == 'search_semantic_scholar':
                            papers = await self.web_search.search_semantic_scholar(
                                query=args.get('query', ''),
                                max_results=args.get('max_results', 10),
                                year_range=args.get('year_range', 5)
                            )
                            result = self.web_search.format_papers_for_context(papers, max_papers=5)
                            tool_results.append({
                                'tool_call_id': tool_call['id'],
                                'output': result
                            })
                            self.log(f"✓ Semantic Scholar search: found {len(papers)} papers")
                    
                    except Exception as e:
                        self.log(f"Tool execution failed ({tool_name}): {e}", "warning")
                        tool_results.append({
                            'tool_call_id': tool_call['id'],
                            'output': f"Error: {str(e)}"
                        })
                
                # Step 4: Continue conversation with tool results
                if tool_results:
                    self.log("Analyzing literature and generating hypotheses...")
                    
                    # Format tool results for LLM
                    formatted_tool_results = []
                    for i, tool_result in enumerate(tool_results):
                        formatted_tool_results.append({
                            "id": response['tool_calls'][i]['id'],
                            "name": response['tool_calls'][i]['name'],
                            "result": tool_result['output']
                        })
                    
                    final_response = await self.llm.continue_with_tool_results(
                        messages=[{"role": "user", "content": initial_message}],
                        tool_calls=response['tool_calls'],
                        tool_results=formatted_tool_results,
                        tools=tools,
                        temperature=0.7,
                        purpose="hypothesis generation from literature"
                    )
                    
                    # Extract content from final response
                    response = final_response
            
            # Step 5: Generate structured hypotheses from LLM response
            if not response.get('content'):
                self.log("No hypotheses generated from literature", "warning")
                return []
            
            # Use structured prompt to format the hypotheses from LLM analysis
            structure_prompt = f"""Based on your literature analysis, format your hypotheses in JSON:

Research Goal: {research_goal.description}
Domain: {research_goal.domain}
Focus Areas: {focus_areas_str}

Provide 3 novel hypotheses informed by the literature you reviewed.
Each hypothesis should:
- Build on existing research gaps
- Propose a testable approach
- Be innovative yet grounded in current knowledge

Format:
{{
    "hypotheses": [
        {{
            "content": "Full hypothesis statement",
            "summary": "One-sentence summary",
            "category": "Type of hypothesis",
            "reasoning": "Why this hypothesis is novel and valuable"
        }}
    ]
}}"""
            
            # Get structured response
            structured_response = await self.llm.generate_json(
                messages=[
                    {"role": "user", "content": initial_message},
                    {"role": "assistant", "content": response['content']},
                    {"role": "user", "content": structure_prompt}
                ],
                temperature=0.7,
                purpose="structuring hypotheses"
            )
            
            # Parse and create Hypothesis objects
            hypotheses = []
            for i, hyp_data in enumerate(structured_response.get("hypotheses", [])):
                # Validate required fields
                if not hyp_data.get("content"):
                    self.log(f"Skipping hypothesis {i}: missing content", "warning")
                    continue
                
                hypothesis = Hypothesis(
                    id=f"hyp_lit_{datetime.now().timestamp()}_{i}",
                    content=hyp_data["content"],
                    category=hyp_data.get("category", "literature-based"),
                    summary=hyp_data.get("summary", hyp_data["content"][:200]),  # Fallback to content prefix
                    generated_at=datetime.now(),
                    supporting_papers=[],  # Papers are embedded in the analysis
                    metadata={
                        "generation_method": "literature_tool_calling",
                        "reasoning": hyp_data.get("reasoning", ""),
                        "analysis": response.get('content', '')[:500]  # Store first 500 chars of analysis
                    }
                )
                hypotheses.append(hypothesis)
            
            self.log(f"Generated {len(hypotheses)} hypotheses from literature")
            return hypotheses
        
        except Exception as e:
            self.log(f"Inner error in literature generation: {e}", "error")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "debug")
            return []
    
    async def generate_from_debates(self, research_goal: ResearchGoal) -> List[Hypothesis]:
        """Generate hypotheses through simulated scientific debates"""
        self.log("Generating hypotheses through simulated debates")
        
        if not self.llm:
            self.log("LLM not configured, skipping debate generation", "warning")
            return []
        
        try:
            # Define debate experts with different perspectives
            experts = [
                {"role": "optimistic_scientist", "perspective": "Focus on potential breakthroughs and possibilities"},
                {"role": "skeptical_reviewer", "perspective": "Critical of methodology, demand evidence"},
                {"role": "practical_methodologist", "perspective": "Emphasize feasibility and implementation"}
            ]
            
            # Run debate rounds
            debate_rounds = 3
            debate_history = []
            current_hypothesis = f"Initial research direction: {research_goal.description}"
            
            for round_num in range(1, debate_rounds + 1):
                self.log(f"  Round {round_num}/{debate_rounds}")
                
                round_discussions = []
                # Prepare focus areas string safely
                focus_areas_str = ""
                if hasattr(research_goal, 'focus_areas') and research_goal.focus_areas:
                    if isinstance(research_goal.focus_areas, list):
                        focus_areas_str = ', '.join(str(x) for x in research_goal.focus_areas)
                    else:
                        focus_areas_str = str(research_goal.focus_areas)
                
                for expert in experts:
                    prompt = f"""You are a {expert['role']} in a scientific debate. {expert['perspective']}

Research Goal: {research_goal.description}
Domain: {research_goal.domain}
Focus Areas: {focus_areas_str}"

Current Hypothesis Under Discussion:
{current_hypothesis}

Previous Debate Points:
{json.dumps(debate_history[-2:], indent=2) if debate_history else "None yet"}

Round {round_num} of {debate_rounds}:
- Critique the current hypothesis from your perspective
- Suggest specific improvements
- Highlight potential issues or strengths

Provide as JSON:
{{
  "critique": "Your critical analysis",
  "suggestions": ["specific suggestion 1", "suggestion 2"],
  "concerns": ["potential issue 1", "issue 2"],
  "support": ["strength 1", "strength 2"]
}}"""
                    
                    response = await self.llm.generate_json(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        purpose="debate expert discussion"
                    )
                    response["expert"] = expert["role"]
                    round_discussions.append(response)
                
                debate_history.append({
                    "round": round_num,
                    "discussions": round_discussions
                })
                
                # Synthesize improvements for next round
                if round_num < debate_rounds:
                    synthesis_prompt = f"""Based on this debate round, synthesize an improved hypothesis.

Debate Discussions:
{json.dumps(round_discussions, indent=2)}

Create an improved version that addresses critiques while maintaining strengths.
Respond with just the improved hypothesis text (2-3 sentences)."""
                    
                    current_hypothesis = await self.llm.generate(
                        messages=[{"role": "user", "content": synthesis_prompt}],
                        temperature=0.6,
                        purpose="debate synthesis"
                    )
            
            # Generate final hypotheses from debate
            # Prompt Caching을 사용하여 전체 debate history 보존하면서 토큰 절약
            final_prompt = f"""Based on this complete scientific debate, generate 2-3 refined research hypotheses.

Research Goal: {research_goal.description}

Complete Debate History:
{json.dumps(debate_history, indent=2)}

Generate hypotheses that:
1. Address concerns raised by skeptics
2. Incorporate practical feasibility
3. Maintain scientific rigor
4. Are testable and specific

Provide as JSON:
{{
  "hypotheses": [
    {{
      "content": "Detailed hypothesis (3-5 sentences)",
      "summary": "One sentence summary",
      "category": "Research category",
      "reasoning": "How debate informed this"
    }}
  ]
}}"""
            
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.6,
                purpose="debate-based hypothesis generation",
                use_cache=True
            )
            
            # Create Hypothesis objects
            hypotheses = []
            for i, hyp_data in enumerate(response.get("hypotheses", [])):
                hypothesis = Hypothesis(
                    id=f"hyp_debate_{datetime.now().timestamp()}_{i}",
                    content=hyp_data["content"],
                    category=hyp_data.get("category", "debate-generated"),
                    summary=hyp_data["summary"],
                    generated_at=datetime.now(),
                    metadata={
                        "generation_method": "debate",
                        "reasoning": hyp_data.get("reasoning", ""),
                        "debate_rounds": debate_rounds
                    }
                )
                hypotheses.append(hypothesis)
                self.memory.store_hypothesis(hypothesis)
            
            self.log(f"  ✅ Generated {len(hypotheses)} hypotheses from debates")
            return hypotheses
            
        except Exception as e:
            self.log(f"Error in debate generation: {e}", "error")
            return []
    
    async def generate_from_assumptions(self, research_goal: ResearchGoal) -> List[Hypothesis]:
        """Generate hypotheses by identifying testable assumptions"""
        self.log("Generating hypotheses from testable assumptions")
        
        if not self.llm:
            self.log("LLM not configured, skipping assumption generation", "warning")
            return []
        
        try:
            # Step 1: Identify key assumptions
            focus_areas_str = ""
            if hasattr(research_goal, 'focus_areas') and research_goal.focus_areas:
                if isinstance(research_goal.focus_areas, list):
                    focus_areas_str = ', '.join(str(x) for x in research_goal.focus_areas)
                else:
                    focus_areas_str = str(research_goal.focus_areas)
            
            assumptions_prompt = f"""Identify testable assumptions for this research goal.

Research Goal: {research_goal.description}
Domain: {research_goal.domain}
Focus Areas: {focus_areas_str}"

Generate 5-7 fundamental assumptions that:
1. If proven true, would significantly advance the research goal
2. Are testable through experiments
3. Build on each other logically
4. Cover different aspects of the problem

Provide as JSON:
{{
  "assumptions": [
    {{
      "assumption": "Clear statement",
      "rationale": "Why this is important",
      "testability": "How to test this",
      "dependency": "independent or depends on assumption X"
    }}
  ]
}}"""
            
            assumptions_response = await self.llm.generate_json(
                messages=[{"role": "user", "content": assumptions_prompt}],
                temperature=0.7,
                purpose="assumption identification"
            )
            
            assumptions = assumptions_response.get("assumptions", [])
            self.log(f"  💡 Identified {len(assumptions)} key assumptions")
            
            # Step 2: For top assumptions, identify sub-assumptions
            detailed_assumptions = []
            for assumption in assumptions[:5]:  # Limit to top 5
                sub_prompt = f"""Break down this assumption into specific sub-assumptions.

Main Assumption: {assumption['assumption']}
Rationale: {assumption['rationale']}

Identify 2-3 sub-assumptions that:
1. Are more specific and testable
2. Together support the main assumption
3. Can be validated independently

Provide as JSON:
{{
  "sub_assumptions": [
    {{
      "sub_assumption": "Specific statement",
      "validation_method": "How to test",
      "criticality": "critical/important/supporting"
    }}
  ]
}}"""
                
                sub_response = await self.llm.generate_json(
                    messages=[{"role": "user", "content": sub_prompt}],
                    temperature=0.6,
                    purpose="assumption validation"
                )
                
                detailed_assumptions.append({
                    "main": assumption,
                    "sub_assumptions": sub_response.get("sub_assumptions", [])
                })
            
            # Step 3: Aggregate into complete hypotheses
            hypothesis_prompt = f"""Create research hypotheses by aggregating these assumptions.

Research Goal: {research_goal.description}

Detailed Assumptions:
{json.dumps(detailed_assumptions, indent=2)}

Generate 2-3 complete hypotheses that:
1. Combine multiple assumptions into testable proposals
2. Include specific experimental approaches
3. Explain how assumptions lead to the hypothesis
4. Prioritize critical assumptions

Provide as JSON:
{{
  "hypotheses": [
    {{
      "content": "Complete hypothesis with methodology",
      "summary": "One sentence summary",
      "category": "Category",
      "key_assumptions": ["assumption 1", "assumption 2"],
      "experimental_approach": "How to test this"
    }}
  ]
}}"""
            
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": hypothesis_prompt}],
                temperature=0.6,
                purpose="assumption-based hypothesis generation"
            )
            
            # Create Hypothesis objects
            hypotheses = []
            for i, hyp_data in enumerate(response.get("hypotheses", [])):
                hypothesis = Hypothesis(
                    id=f"hyp_assume_{datetime.now().timestamp()}_{i}",
                    content=hyp_data["content"],
                    category=hyp_data.get("category", "assumption-based"),
                    summary=hyp_data["summary"],
                    generated_at=datetime.now(),
                    metadata={
                        "generation_method": "assumptions",
                        "key_assumptions": hyp_data.get("key_assumptions", []),
                        "experimental_approach": hyp_data.get("experimental_approach", "")
                    }
                )
                hypotheses.append(hypothesis)
                self.memory.store_hypothesis(hypothesis)
            
            self.log(f"  ✅ Generated {len(hypotheses)} hypotheses from assumptions")
            return hypotheses
            
        except Exception as e:
            self.log(f"Error in assumption-based generation: {e}", "error")
            return []
    
    async def expand_research_space(
        self,
        research_goal: ResearchGoal,
        existing_hypotheses: List[Hypothesis],
        meta_review_feedback: Optional[Dict]
    ) -> List[Hypothesis]:
        """Expand research space based on existing work"""
        self.log("Expanding research space based on existing hypotheses")
        
        if not self.llm or not existing_hypotheses:
            self.log("Need LLM and existing hypotheses to expand", "warning")
            return []
        
        try:
            # Analyze existing hypotheses to find gaps
            existing_summaries = [h.summary for h in existing_hypotheses[:10]]
            existing_categories = list(set([h.category for h in existing_hypotheses]))
            
            # Get feedback if available
            avoided_patterns = []
            explore_areas = []
            if meta_review_feedback:
                avoided_patterns = meta_review_feedback.get("avoid_patterns", [])
                explore_areas = meta_review_feedback.get("explore_areas", [])
            
            expand_prompt = f"""Identify unexplored research directions for this goal.

Research Goal: {research_goal.description}
Domain: {research_goal.domain}

Existing Hypotheses (sample):
{chr(10).join([f"- {s}" for s in existing_summaries[:7]])}

Existing Categories: {', '.join(existing_categories)}

Patterns to Avoid:
{chr(10).join([f"- {p}" for p in avoided_patterns]) if avoided_patterns else "None specified"}

Areas to Explore:
{chr(10).join([f"- {a}" for a in explore_areas]) if explore_areas else "None specified"}

Identify 3-5 NEW research directions that:
1. Are NOT covered by existing hypotheses
2. Complement existing work
3. Explore different methodologies or approaches
4. Address potential gaps in current hypotheses

Provide as JSON:
{{
  "new_directions": [
    {{
      "direction": "Description of new research direction",
      "rationale": "Why this is unexplored and valuable",
      "methodology": "Suggested approach",
      "category": "Research category"
    }}
  ]
}}"""
            
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": expand_prompt}],
                temperature=0.8,  # Higher for more creativity
                purpose="research space expansion"
            )
            
            # Generate hypotheses for each new direction
            hypotheses = []
            for i, direction in enumerate(response.get("new_directions", [])):
                
                hyp_prompt = f"""Create a detailed research hypothesis for this direction.

Research Goal: {research_goal.description}

New Direction: {direction['direction']}
Rationale: {direction['rationale']}
Methodology: {direction['methodology']}

Create a specific, testable hypothesis that:
1. Addresses this new direction
2. Is distinct from existing work
3. Includes concrete methodology
4. Is scientifically rigorous

Provide as JSON:
{{
  "content": "Detailed hypothesis (3-5 sentences)",
  "summary": "One sentence summary"
}}"""
                
                hyp_response = await self.llm.generate_json(
                    messages=[{"role": "user", "content": hyp_prompt}],
                    temperature=0.7,
                    purpose="expansion hypothesis generation"
                )
                
                hypothesis = Hypothesis(
                    id=f"hyp_expand_{datetime.now().timestamp()}_{i}",
                    content=hyp_response["content"],
                    category=direction.get("category", "exploratory"),
                    summary=hyp_response["summary"],
                    generated_at=datetime.now(),
                    metadata={
                        "generation_method": "expansion",
                        "new_direction": direction["direction"],
                        "rationale": direction["rationale"]
                    }
                )
                hypotheses.append(hypothesis)
                self.memory.store_hypothesis(hypothesis)
            
            self.log(f"Expanded research space with {len(hypotheses)} new hypotheses")
            return hypotheses
            
        except Exception as e:
            self.log(f"Error expanding research space: {e}", "error")
            return []
