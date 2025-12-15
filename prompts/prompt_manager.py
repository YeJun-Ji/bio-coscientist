"""
Prompt Manager - Templates for LLM interactions
"""

from typing import Dict, Any, Optional


class PromptManager:
    """Manages prompt templates for various research tasks"""
    
    def __init__(self):
        """Initialize prompt templates"""
        self.templates = {}
    
    def get(self, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt template and format it with provided arguments
        
        Args:
            prompt_name: Name of the prompt template
            **kwargs: Variables to format into the template
            
        Returns:
            Formatted prompt string
        """
        template = self.templates.get(prompt_name)
        if not template:
            # Return a default prompt if specific template not found
            return self._get_default_prompt(prompt_name, **kwargs)
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable for prompt '{prompt_name}': {e}")
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Alias for get() method for backward compatibility"""
        return self.get(prompt_name, **kwargs)
    
    def _get_default_prompt(self, prompt_name: str, **kwargs) -> str:
        """Generate default prompts for common use cases"""
        defaults = {
            "hypothesis_generation": "Generate novel research hypotheses for: {goal}\nFocus on: {focus_areas}\nDomain: {domain}",
            "generation_literature": """You are an expert researcher generating novel hypotheses for a research goal.

Research Goal: {research_goal}
Domain: {domain}
Focus Areas: {focus_areas}

Generate {num_hypotheses} innovative and testable research hypotheses. Each hypothesis should be novel, scientifically rigorous, and actionable.

Respond in JSON format:
{{
  "hypotheses": [
    {{
      "content": "Detailed description of the hypothesis including methodology and expected outcomes",
      "summary": "Brief one-sentence summary",
      "category": "Category of the hypothesis (e.g., experimental, computational, theoretical)",
      "reasoning": "Scientific reasoning and motivation behind this hypothesis"
    }}
  ]
}}

Requirements:
- Each hypothesis must be specific and testable
- Include concrete methodology or approach
- Build on current scientific knowledge
- Propose novel insights or directions
- Consider practical feasibility""",
            "literature_search": "Search for research papers related to: {query}",
            "hypothesis_review": "Review this hypothesis for scientific validity: {hypothesis}",
            "ranking_comparison": "Compare these two hypotheses and determine which is better: A) {hypothesis_a} B) {hypothesis_b}",
            "reflection_initial": """Review this research hypothesis for initial assessment:

{hypothesis}

Please evaluate and provide JSON response with the following structure:
{{
  "correctness": {{
    "score": 0.8,
    "details": "Assessment of scientific accuracy and logical soundness"
  }},
  "quality": {{
    "score": 0.7,
    "details": "Assessment of research quality and methodology"
  }},
  "novelty": {{
    "score": 0.9,
    "details": "Assessment of novelty and innovation"
  }},
  "pass": true,
  "summary": "Brief summary of the assessment"
}}

Focus on scientific accuracy, research quality, and novelty. Provide detailed reasoning for each score (0-1 scale).""",
            "reflection_full": """Conduct a comprehensive review of this research hypothesis:

{hypothesis}

Please provide a detailed JSON assessment:
{{
  "correctness": {{
    "score": 0.8,
    "details": "Detailed analysis of scientific accuracy"
  }},
  "quality": {{
    "score": 0.7,
    "details": "Assessment of methodology and research quality"
  }},
  "novelty": {{
    "score": 0.9,
    "details": "Evaluation of innovation and uniqueness"
  }},
  "feasibility": {{
    "score": 0.6,
    "details": "Practical feasibility assessment"
  }},
  "strengths": ["List of key strengths"],
  "weaknesses": ["List of potential weaknesses"],
  "suggestions": ["Improvement suggestions"],
  "pass": true,
  "confidence": 0.85
}}

Provide thorough analysis with specific reasoning for each dimension."""
        }
        
        template = defaults.get(prompt_name, f"Process this request: {kwargs.get('goal', 'No goal specified')}")
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    def add_template(self, name: str, template: str):
        """Add or update a prompt template"""
        self.templates[name] = template
    
    def list_templates(self):
        """List all available prompt templates"""
        return list(self.templates.keys())
