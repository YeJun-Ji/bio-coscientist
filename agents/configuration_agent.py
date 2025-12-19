"""
Configuration Agent - Parses research goal and creates research plan configuration

This agent analyzes the research goal provided by the scientist and generates
a comprehensive research plan configuration that guides the entire research process.

Option B Implementation: Restructured configuration based on AI Co-Scientist paper (Section 3.2)
- Static vs Dynamic constraints separation
- Pipeline functions extracted from problem statement
- Per-hypothesis proposed_constraints support
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..external_apis import LLMClient
from ..core import ResearchGoal, ParsedProblem, Requirement
from ..prompts import PromptManager

logger = logging.getLogger(__name__)


class ConfigurationAgent:
    """
    Analyzes research goals and generates research plan configurations.

    This agent uses LLM to parse natural language research goals and extract:
    - parsed_goal: Research objective with domain classification
    - constraints: Static and Dynamic constraints
    - evaluation_criteria: Pipeline functions + quality metrics
    - workflow: Execution targets and convergence criteria
    - requirements: Research requirements that hypotheses must address

    Option B Structure:
    - constraints.static: All hypotheses must satisfy (parsed from problem)
    - constraints.dynamic: Each hypothesis proposes specific values
    - evaluation_criteria.pipeline_functions: From problem's required steps

    Backward Compatibility Fields (kept for existing code):
    - research_objective, workflow_strategy, agent_priorities
    - constraints_legacy, research_steps
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.logger = logging.getLogger("ConfigurationAgent")

        # Initialize prompt manager for template-based prompts
        self.prompt_manager = PromptManager()
    
    async def create_configuration(
        self,
        research_goal: ResearchGoal,
        user_preferences: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create SIMPLIFIED research plan configuration (v3.0).

        This creates a minimal, clean configuration with only essential fields:
        1. original_research_goal - Preserves original problem text
        2. parsed_problem - Requirements and execution order
        3. constraints - Simple list of hard constraints
        4. execution_plan - Convergence criteria and agent config
        5. metadata - Tracking info

        Args:
            research_goal: ResearchGoal object
            user_preferences: Optional user-specified preferences
            save_path: Optional path to save the configuration JSON file

        Backward compatibility fields (parsed_goal, research_objective) are
        automatically added for existing code.

        Args:
            research_goal: The research objective
            user_preferences: Optional user-specified preferences

        Returns:
            Simplified research plan configuration dict (~70% smaller than v2.0)
        """

        try:
            # Step 1: Parse problem structure to extract requirements
            self.logger.info("📋 Parsing problem structure to extract requirements...")
            parsed_problem = await self._parse_problem_structure(research_goal.description)
            self.logger.info(f"✓ Extracted {len(parsed_problem.requirements)} requirements "
                           f"(type: {parsed_problem.problem_type}, format: {parsed_problem.format_detected})")

            # Step 2: Extract constraints as simple list (NEW v3.0)
            self.logger.info("🔒 Extracting hard constraints...")
            constraints = await self._extract_constraints_simple(research_goal.description)
            self.logger.info(f"✓ Extracted {len(constraints)} hard constraints")

            # Step 3: Build execution plan (NEW v3.0)
            self.logger.info("📊 Building execution plan...")
            execution_order = parsed_problem.get_execution_order()
            execution_plan = self._build_execution_plan(
                execution_order=execution_order,
                user_preferences=user_preferences
            )
            self.logger.info(f"✓ Execution plan: {len(execution_order)} groups, "
                           f"{execution_plan['per_requirement']['max_iterations']} max iterations")

            # Step 4: Store original research goal (NEW v3.0)
            original_goal = {
                "raw_text": research_goal.description,
                "title": parsed_problem.title,
                "source_file": research_goal.metadata.get("source_file", "unknown")
            }

            # Step 5: Build simplified config
            parsed_problem_dict = parsed_problem.to_dict()
            # Add execution_order to parsed_problem dict (not included by to_dict())
            parsed_problem_dict["execution_order"] = execution_order

            config = {
                "original_research_goal": original_goal,
                "parsed_problem": parsed_problem_dict,
                "constraints": constraints,
                "execution_plan": execution_plan,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "goal_id": research_goal.goal_id,
                    "version": "3.0",
                    "config_format": "simplified"
                }
            }

            # Log summary
            self._log_requirements(parsed_problem)
            self.logger.info(f"✅ Configuration created (v3.0 simplified format)")

            # Save to file if save_path provided
            if save_path:
                self._save_configuration(config, save_path)

            return config

        except Exception as e:
            self.logger.error(f"Failed to create configuration: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Fallback to default configuration
            return self._create_default_configuration(research_goal)

    async def _parse_problem_structure(self, problem_text: str) -> ParsedProblem:
        """
        Parse problem text to extract requirements that hypotheses must answer.

        Extracts:
        1. Background (배경/가설)
        2. Input Data (입력 데이터) - optional
        3. Requirements (문제가 요구하는 것들) - each (1), (2), etc. becomes a requirement

        The key insight: Each numbered item in the problem is a REQUIREMENT
        that a good hypothesis must answer well.

        Supports multiple formats:
        - Flat parenthesized: (1), (2), ...
        - Flat dot-number: 1., 2., ...
        - Hierarchical: (A) + sub-requirements
        - Semi-hierarchical: (A), (B) without sub-numbering

        Args:
            problem_text: Raw problem description text

        Returns:
            ParsedProblem object with requirements that hypotheses must address
        """
        # Build prompt using template
        prompt = self.prompt_manager.get_prompt(
            "configuration/problem_structure_parsing",
            problem_text=problem_text
        )

        try:
            # Call LLM to parse structure
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for more consistent parsing
                purpose="problem_structure_parsing"
            )

            # Convert response to ParsedProblem
            return self._response_to_parsed_problem(response)

        except Exception as e:
            self.logger.warning(f"Failed to parse problem structure: {e}, using fallback")
            return self._create_fallback_parsed_problem(problem_text)

    def _response_to_parsed_problem(self, response: Dict[str, Any]) -> ParsedProblem:
        """Convert LLM response to ParsedProblem object."""
        # Parse requirements (support both 'requirements' and legacy 'research_steps')
        requirements = []
        req_data = response.get("requirements", response.get("research_steps", []))
        for data in req_data:
            req = Requirement(
                requirement_id=data.get("requirement_id", data.get("step_id", "")),
                title=data.get("title", ""),
                description=data.get("description", ""),
                parent_id=data.get("parent_id"),
                expected_deliverables=data.get("expected_deliverables", data.get("required_deliverables", [])),
                depends_on=data.get("depends_on", []),
                can_parallelize=data.get("can_parallelize", True),
                order=data.get("order", 0),
                requirement_type=data.get("requirement_type", data.get("step_type", "answer")),
                priority=data.get("priority", 1)
            )
            requirements.append(req)

        return ParsedProblem(
            title=response.get("title", "Untitled Problem"),
            background=response.get("background", ""),
            input_data_description=response.get("input_data_description"),
            requirements=requirements,
            problem_type=response.get("problem_type", "flat"),
            major_sections=response.get("major_sections", {}),
            format_detected=response.get("format_detected", "flat_paren_num")
        )

    def _create_fallback_parsed_problem(self, problem_text: str) -> ParsedProblem:
        """Create a basic ParsedProblem when LLM parsing fails."""
        import re

        # Try to detect format and extract requirements using regex
        requirements = []

        # Pattern 1: (1), (2), etc.
        paren_num_pattern = r'\((\d+)\)\s*(.+?)(?=\(\d+\)|$)'
        matches = re.findall(paren_num_pattern, problem_text, re.DOTALL)

        if matches:
            format_detected = "flat_paren_num"
            for i, (num, content) in enumerate(matches):
                requirements.append(Requirement(
                    requirement_id=num,
                    title=content.strip()[:50] + "..." if len(content) > 50 else content.strip(),
                    description=content.strip(),
                    order=i
                ))
        else:
            # Pattern 2: 1., 2., etc.
            dot_num_pattern = r'^(\d+)\.\s*(.+?)(?=^\d+\.|$)'
            matches = re.findall(dot_num_pattern, problem_text, re.MULTILINE | re.DOTALL)

            if matches:
                format_detected = "flat_dot_num"
                for i, (num, content) in enumerate(matches):
                    requirements.append(Requirement(
                        requirement_id=num,
                        title=content.strip()[:50],
                        description=content.strip(),
                        order=i
                    ))
            else:
                format_detected = "unknown"
                # Create a single requirement for the entire problem
                requirements.append(Requirement(
                    requirement_id="1",
                    title="Research Task",
                    description=problem_text[:500],
                    order=0
                ))

        return ParsedProblem(
            title="Research Problem",
            background=problem_text[:1000],  # First 1000 chars as background
            requirements=requirements,
            problem_type="flat",
            format_detected=format_detected
        )

    def _log_requirements(self, parsed_problem: ParsedProblem) -> None:
        """Log parsed requirements for debugging."""
        self.logger.info(f"Problem: {parsed_problem.title}")
        self.logger.info(f"Type: {parsed_problem.problem_type}, Format: {parsed_problem.format_detected}")

        if parsed_problem.major_sections:
            self.logger.info(f"Major sections: {list(parsed_problem.major_sections.keys())}")

        for req in parsed_problem.requirements:
            deps = f" (depends: {req.depends_on})" if req.depends_on else ""
            parent = f" [parent: {req.parent_id}]" if req.parent_id else ""
            self.logger.info(f"  Requirement {req.requirement_id}: {req.title}{parent}{deps}")
    
    def _create_default_configuration(self, research_goal: ResearchGoal) -> Dict[str, Any]:
        """
        Create a default configuration if LLM fails (v3.0 simplified structure).

        This is a fallback that provides minimal viable configuration.
        """
        self.logger.warning("Using default configuration (v3.0)")

        # Create minimal ParsedProblem
        from ..core import ParsedProblem, Requirement

        parsed_problem = ParsedProblem(
            title=f"Research: {research_goal.description[:100]}...",
            background="Default research problem",
            requirements=[
                Requirement(
                    requirement_id="1",
                    title="Main research objective",
                    description=research_goal.description,
                    depends_on=[],
                    expected_deliverables=["Research findings"]
                )
            ],
            problem_type="hypothesis_generation"
        )

        # Get execution order from parsed_problem
        execution_order = parsed_problem.get_execution_order()

        # Build parsed_problem dict with execution_order
        parsed_problem_dict = parsed_problem.to_dict()
        parsed_problem_dict["execution_order"] = execution_order

        # Build simplified config
        config = {
            "original_research_goal": {
                "raw_text": research_goal.description,
                "title": parsed_problem.title,
                "source_file": "default"
            },
            "parsed_problem": parsed_problem_dict,
            "constraints": ["Should be scientifically valid"],
            "execution_plan": {
                "execution_order": execution_order,
                "per_requirement": {
                    "max_iterations": 5
                },
                "convergence": {
                    "min_quality_score": 0.7,
                    "min_elo_rating": 1300,
                    "deliverables_ratio": 0.8
                },
                "agent_config": {
                    "ranking": {"elo_k_factor": 32},
                    "evolution": {"methods": ["grounding", "coherence"]}
                }
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "goal_id": research_goal.goal_id,
                "version": "3.0",
                "config_format": "simplified"
            }
        }

        return config

    async def _extract_constraints_simple(self, problem_text: str) -> List[str]:
        """
        Extract only hard constraints as flat list.

        Unlike the old method that created complex nested structures,
        this extracts constraints as a simple list for the simplified config.

        Args:
            problem_text: Research problem text

        Returns:
            List of hard constraint strings
        """
        prompt = f"""Extract HARD CONSTRAINTS from this research problem.
Return only concrete, non-negotiable requirements.

**IMPORTANT INSTRUCTIONS:**
1. Extract ONLY hard constraints (must-have requirements)
2. Each constraint should be specific and measurable
3. Do NOT include soft preferences or nice-to-haves
4. Return as simple list of strings

Problem:
{problem_text}

Return valid JSON:
{{
  "hard_constraints": [
    "constraint 1",
    "constraint 2",
    ...
  ]
}}
"""

        try:
            response = await self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                purpose="extract_constraints_simple"
            )

            constraints = response.get("hard_constraints", [])
            self.logger.info(f"Extracted {len(constraints)} hard constraints")
            return constraints

        except Exception as e:
            self.logger.warning(f"Failed to extract constraints: {e}")
            return []

    def _build_execution_plan(
        self,
        execution_order: List[List[str]],
        user_preferences: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Build execution plan from user preferences.

        This replaces the old 'workflow' field with a clearer execution strategy.

        Args:
            execution_order: Topologically sorted requirement groups
            user_preferences: Optional user-specified preferences

        Returns:
            Execution plan dict with convergence criteria and agent config
        """
        prefs = user_preferences or {}

        execution_plan = {
            "execution_order": execution_order,
            "per_requirement": {
                "max_iterations": prefs.get("max_iterations_per_requirement", 5)
            },
            "convergence": {
                "min_quality_score": prefs.get("min_quality_score", 0.7),
                "min_elo_rating": prefs.get("min_elo_rating", 1300),
                "deliverables_ratio": prefs.get("deliverables_ratio", 0.8)
            },
            "agent_config": {
                "ranking": {
                    "elo_k_factor": prefs.get("elo_k_factor", 32)
                },
                "evolution": {
                    "methods": prefs.get("evolution_methods", ["grounding", "coherence"])
                }
            }
        }

        self.logger.info(f"Built execution plan: {len(execution_order)} groups, "
                        f"max_iterations={execution_plan['per_requirement']['max_iterations']}")

        return execution_plan

    def _save_configuration(self, config: Dict[str, Any], save_path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary to save
            save_path: Path to save the configuration file (directory or full path)
        """
        import json
        from pathlib import Path

        save_path_obj = Path(save_path)

        # If save_path is a directory, create config.json inside it
        if save_path_obj.is_dir():
            config_file = save_path_obj / "config.json"
        else:
            # If save_path is a file path, use it directly
            config_file = save_path_obj
            # Ensure parent directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)

        # Save configuration with pretty formatting
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)

            # Calculate file size
            file_size = config_file.stat().st_size / 1024  # KB

            self.logger.info(f"📝 Configuration saved to: {config_file}")
            self.logger.info(f"   Size: {file_size:.1f} KB")

            # Log version info
            version = config.get("metadata", {}).get("version", "unknown")
            config_format = config.get("metadata", {}).get("config_format", "unknown")
            self.logger.info(f"   Version: {version} ({config_format})")

        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise
