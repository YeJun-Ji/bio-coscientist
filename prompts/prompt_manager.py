"""
Prompt Manager - Templates for LLM interactions with Jinja2 support
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

try:
    from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Template = None

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates with Jinja2 support for file-based templates"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize prompt manager
        
        Args:
            template_dir: Directory containing .jinja2 template files
                         Defaults to biocoscientist/prompts/templates/
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for compiled templates
        self._template_cache: Dict[str, Any] = {}
        
        # In-memory templates (for backward compatibility and fallback)
        self.templates: Dict[str, str] = {}
        
        # Initialize Jinja2 environment if available
        if JINJA2_AVAILABLE:
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False
            )
        else:
            self.env = None
            self.logger.warning("Jinja2 not available. Install with: pip install jinja2")
    
    def get_prompt(self, prompt_name: str, problem_type: Optional[str] = None, **kwargs) -> str:
        """
        Get and render a prompt template
        
        Args:
            prompt_name: Template name (e.g., "generation_data_based" or "generation/data_based")
            problem_type: Optional problem type for specialized templates (e.g., "protein_binder_design")
            **kwargs: Variables to inject into template
        
        Returns:
            Rendered prompt string
        """
        # Try problem-specific template first (if problem_type provided)
        if problem_type and self.env:
            specific_name = f"{prompt_name}_{problem_type}"
            template = self._load_template(specific_name)
            if template:
                try:
                    return template.render(**kwargs)
                except Exception as e:
                    self.logger.error(f"Error rendering template {specific_name}: {e}")
        
        # Try general template from file
        if self.env:
            template = self._load_template(prompt_name)
            if template:
                try:
                    return template.render(**kwargs)
                except Exception as e:
                    self.logger.error(f"Error rendering template {prompt_name}: {e}")
        
        # Fallback to in-memory templates
        if prompt_name in self.templates:
            template_str = self.templates[prompt_name]
            try:
                if JINJA2_AVAILABLE and Template:
                    return Template(template_str).render(**kwargs)
                else:
                    return template_str.format(**kwargs)
            except Exception as e:
                self.logger.error(f"Error rendering in-memory template {prompt_name}: {e}")
        
        # Last resort: default prompts
        return self._get_default_prompt(prompt_name, **kwargs)
    
    def _load_template(self, template_name: str) -> Optional[Any]:
        """
        Load a template from file or cache
        
        Args:
            template_name: Template name (can include directory, e.g., "generation/data_based")
        
        Returns:
            Jinja2 Template object or None
        """
        if not self.env:
            return None
        
        # Check cache
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Try with .jinja2 extension
        template_path = f"{template_name}.jinja2"
        try:
            template = self.env.get_template(template_path)
            self._template_cache[template_name] = template
            self.logger.debug(f"Loaded template: {template_path}")
            return template
        except TemplateNotFound:
            self.logger.debug(f"Template not found: {template_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading template {template_path}: {e}")
            return None
    
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
    