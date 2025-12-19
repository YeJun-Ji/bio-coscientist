"""
BioCoScientist - Main interface for Bio AI Research Assistant
Supports 5 problem types with specialized handlers
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 로깅 설정 - 파일과 콘솔 모두에 출력
def setup_logging(experiment_name: Optional[str] = None):
    """
    Set up dual logging configuration (console + full terminal log file).

    Logs are saved to:
    - logs/<project_name_timestamp>/full_terminal.log - All terminal output
    - logs/<project_name_timestamp>/supervisor.log - Essential flow only (managed by SupervisorAgent)

    Args:
        experiment_name: Name of the project (used as directory name in logs/)

    Returns:
        Path: Session directory path (logs/<project_name_timestamp>/)
    """
    # Create session directory: logs/<project_name_timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not experiment_name:
        experiment_name = f"experiment_{timestamp}"
    else:
        # Add timestamp to experiment name
        experiment_name = f"{experiment_name}_{timestamp}"

    # Session directory: logs/<project_name_timestamp>/
    session_dir = Path("./logs") / experiment_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create RAs subdirectory for RequirementAnswer configs
    ras_dir = session_dir / "RAs"
    ras_dir.mkdir(exist_ok=True)

    # ========== Dual logging: Console + Full Terminal Log ==========
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Clear existing handlers

    # Console handler - show progress to user
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - save all terminal output to full_terminal.log
    terminal_log_file = session_dir / "full_terminal.log"
    file_handler = logging.FileHandler(terminal_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # ========== Suppress noisy external libraries ==========
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)

    return session_dir

# Support both relative imports (when used as module) and absolute imports (when run directly)
try:
    from .core import ResearchGoal
    from .agents import SupervisorAgent
    from .tools.registry import ToolRegistry
    from .prompts.prompt_manager import PromptManager
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from biocoscientist.core import ResearchGoal
    from biocoscientist.agents import SupervisorAgent
    from biocoscientist.tools.registry import ToolRegistry
    from biocoscientist.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class BioCoScientist:
    """
    Bio AI Co-Scientist - Problem-Agnostic Biomedical Research Assistant
    
    New Architecture:
    - Dynamic research planning based on LLM analysis of research goals
    - Adaptive task generation and worker management
    - No predefined problem types - handles any biomedical research question
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, session_dir: Optional[Path] = None):
        """Initialize the Bio AI Co-Scientist system"""
        self.config = config or self.default_config()
        
        # Store session directory for logging
        self.session_dir = session_dir
        if session_dir:
            self.config["session_dir"] = str(session_dir)
        
        # Initialize new supervisor
        self.supervisor = SupervisorAgent(self.config)
        self.logger = logging.getLogger("BioCoScientist")
    
    @staticmethod
    def default_config() -> Dict[str, Any]:
        """Return default configuration"""
        # OpenRouter API 키 읽기
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("⚠️  경고: OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")
            print("   .env 파일에 OPENROUTER_API_KEY=your_key_here 를 추가하세요.")
        
        return {
            "storage_path": "./research_memory",
            "llm": {
                "provider": os.getenv("LLM_PROVIDER", "openrouter"),
                "model": os.getenv("LLM_MODEL", "anthropic/claude-4.5-sonnet"),
                "api_key": api_key,  # 환경변수에서 직접 읽기
                "temperature": 0.7,
                "max_tokens": 8192
            },
            "generation": {
                "techniques": ["data", "assumptions", "expansion"]
            },
            "reflection": {
                "review_types": ["initial", "full", "deep_verification", "observation", "simulation"]
            },
            "ranking": {
                "elo_k_factor": 32,
                "initial_rating": 1200
            },
            "proximity": {
                "similarity_threshold": 0.7
            },
            "evolution": {
                "methods": ["grounding", "coherence", "combination", "simplification", "divergent"]
            },
            "meta_review": {
                "overview_format": "standard"
            }
        }

    async def research_from_file(self, problem_file: str, **kwargs) -> Dict[str, Any]:
        """Start research directly from a problem file.
        
        Args:
            problem_file: Path to the problem description file
            **kwargs: Additional research parameters (max_iterations, etc.)
        
        Returns:
            Research results
        """
        # Read problem file
        with open(problem_file, 'r', encoding='utf-8') as f:
            problem_text = f.read()
        
        # Create a simple research goal - ConfigurationAgent will do the detailed parsing
        research_goal = ResearchGoal(
            description=problem_text.strip(),
            domain="Biomedical Research",
            focus_areas=[],  # ConfigurationAgent will extract these
            constraints={},
            success_criteria=[],
            metadata={"source_file": problem_file}
        )
        
        # Run Sequential Confirmation research
        user_preferences = {
            "max_iterations": kwargs.get("max_iterations", 3)
        }

        results = await self.supervisor.run_sequential_confirmation(
            research_goal=research_goal,
            user_preferences=user_preferences
        )

        return results

    def get_confirmed_answers(self) -> Dict[str, Any]:
        """Get all confirmed RequirementAnswers"""
        return self.supervisor.memory.get_all_confirmed_answers()

    def get_best_answers(self) -> Dict[str, Any]:
        """Get the best answer for each requirement"""
        return self.supervisor.memory.get_best_answer_per_requirement()
    
    async def parse_problem_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a problem description file and extract research parameters dynamically.
        
        Args:
            file_path: Path to the problem description file
        
        Returns:
            Dictionary with goal_description, domain, focus_areas, constraints, success_criteria
        """
        self.logger.info(f"Parsing problem file: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            problem_text = f.read()
        
        # Use LLM to extract structured information
        try:
            from .external_apis import LLMClient
        except ImportError:
            from biocoscientist.external_apis import LLMClient
        
        llm_config = self.config.get("llm", {})
        
        # API 키 가져오기 - config에서 이미 환경변수 읽어서 저장됨
        api_key = llm_config.get("api_key")
        if not api_key:
            self.logger.warning("No API key available, using basic parsing")
            return {
                "goal_description": problem_text.strip(),
                "domain": "Biomedical Research",
                "focus_areas": ["Analysis", "Research"],
                "constraints": {},
                "success_criteria": ["Complete analysis"],
                "metadata": {}
            }
        
        try:
            llm_client = LLMClient(
                provider=llm_config.get("provider"),
                model=llm_config.get("model"),
                api_key=api_key,
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 8192)
            )
        except Exception as e:
            print(f"[DEBUG] LLMClient creation failed: {e}")
            raise
        
        extraction_prompt = f"""
당신은 생물의학 연구 문제를 분석하는 전문가입니다.
아래 문제 설명을 읽고, BioCoScientist 시스템에 입력할 구조화된 정보를 추출하세요.

문제 설명:
{problem_text}

다음 형식의 JSON으로 응답하세요:
{{
  "goal_description": "연구 목표를 1-2문장으로 요약",
  "domain": "연구 도메인 (예: Protein Engineering, Drug Discovery, Systems Biology 등)",
  "focus_areas": ["구체적인 연구 영역 1", "구체적인 연구 영역 2", ...],
  "constraints": {{
    "제약조건 키1": "값1",
    "제약조건 키2": "값2"
  }},
  "success_criteria": ["성공 기준 1", "성공 기준 2", ...]
}}

주의사항:
- goal_description은 핵심 목표만 간결하게
- focus_areas는 3-5개 정도의 구체적인 영역
- constraints는 문제에서 명시된 제약조건이나 요구사항
- success_criteria는 보고서에 포함될 내용이 다 포함되었는지 확인하는 기준
"""
        
        try:
            result = await llm_client.generate_json(
                messages=[{"role": "user", "content": extraction_prompt}],
                system="You are an expert in biomedical research problem analysis. Extract structured information accurately."
            )
            
            # Ensure metadata field exists
            if "metadata" not in result:
                result["metadata"] = {}
            
            self.logger.info("Successfully parsed problem file")
            self.logger.debug(f"Extracted: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse problem file with LLM: {e}")
            # Fallback to basic extraction
            return {
                "goal_description": problem_text.strip(),
                "domain": "Biomedical Research",
                "focus_areas": ["Analysis", "Research"],
                "constraints": {},
                "success_criteria": ["Complete analysis"],
                "metadata": {}
            }
    
    def export_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Export research results to file with full memory data for report generation"""
        import json
        from pathlib import Path

        # Get full memory data for comprehensive report generation
        memory_data = self.supervisor.memory.export_to_dict()

        # Convert results to exportable format
        export_data = {
            "research_config": results.get("research_config", {}),
            "final_metrics": results.get("final_metrics", {}),
            "top_hypotheses": results.get("top_hypotheses", []),
            "execution_stats": results.get("execution_stats", {}),
            "research_goal": {
                "description": getattr(self, 'research_goal', ResearchGoal(description="")).description,
                "domain": getattr(self, 'research_goal', ResearchGoal(description="")).domain
            },
            # Full memory data for ReportGenerator
            "hypotheses": memory_data.get("hypotheses", []),
            "reviews": memory_data.get("reviews", []),
            "overviews": memory_data.get("overviews", []),
            "tournament_matches": memory_data.get("tournament_matches", []),
            "meta_reviews": memory_data.get("meta_reviews", [])
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Results exported to {output_path}")
        self.logger.info(f"  - {len(export_data['hypotheses'])} hypotheses")
        self.logger.info(f"  - {len(export_data['reviews'])} reviews")
        self.logger.info(f"  - {len(export_data['overviews'])} overviews")
        
        # 보고서 자동 생성
        try:
            from biocoscientist.utils.report_generator import (
                generate_final_research_report,
                generate_report_from_json
            )

            # reports 폴더 경로 생성
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 파일명 형식: BioCoScientist_Report_research_{timestamp}
            report_name = f"BioCoScientist_Report_research_{timestamp}"

            # ★ NEW: 최고 가설 중심 연구 보고서 생성 (Markdown)
            # results dict에서 직접 best_hypothesis 등 확장 데이터 사용
            final_report_path = str(reports_dir / f"{report_name}_FINAL.md")
            generate_final_research_report(results, final_report_path)
            self.logger.info(f"Final research report generated: {final_report_path}")

            # Legacy: 통계 중심 보고서 (기존 형식 유지)
            legacy_report_path = str(reports_dir / f"{report_name}_statistics.txt")
            generate_report_from_json(output_path, legacy_report_path, "full")
            self.logger.info(f"Statistics report generated: {legacy_report_path}")

            print(f"\n📊 Reports Generated:")
            print(f"  - 📄 Final Research Report: {final_report_path}")
            print(f"  - 📈 Statistics Report: {legacy_report_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
            print(f"⚠️  Report generation failed: {e}")


# ============================================================================
# Main Execution
# ============================================================================

async def main(problem_file: str, session_dir: Path):
    """
    Main execution function - simplified to just run research from file

    Args:
        problem_file: Path to problem description file
        session_dir: Session directory for logs (created by setup_logging)
    """
    # Create BioCoScientist instance with provided session_dir
    bio_coscientist = BioCoScientist(session_dir=session_dir)
    
    # Run research from file - ConfigurationAgent handles all parsing
    results = await bio_coscientist.research_from_file(problem_file)
    
    # Display results
    print("\n" + "="*80)
    print("✅ RESEARCH COMPLETE")
    print("="*80)
    
    final_metrics = results.get('final_metrics', {})
    execution_stats = results.get('execution_stats', {})
    
    print(f"\n📊 Research Metrics:")
    print(f"  Total Hypotheses: {final_metrics.get('total_hypotheses', 0)}")
    print(f"  Reviewed: {final_metrics.get('reviewed_hypotheses', 0)}")
    print(f"  Passed Review: {final_metrics.get('passed_hypotheses', 0)}")
    print(f"  Average ELO: {final_metrics.get('avg_elo_rating', 0):.1f}")
    
    print(f"\n⚙️ Execution Stats:")
    print(f"  Iterations: {execution_stats.get('iterations', 0)}")
    print(f"  Duration: {execution_stats.get('duration_seconds', 0):.1f}s")
    
    # Export results
    bio_coscientist.export_results(results, "research_results.json")


if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python biocoscientist.py <problem_file.txt>")
        print("   Example: python biocoscientist.py problems/minibinder_design.txt")
        sys.exit(1)
    
    problem_file = sys.argv[1]
    
    # Extract project name from problem file and setup logging
    project_name = Path(problem_file).stem
    session_dir = setup_logging(experiment_name=project_name)

    print(f"\n📁 Project: {project_name}")
    print(f"📂 Session directory: {session_dir}")
    print(f"📝 Logs:")
    print(f"   - Full terminal: {session_dir / 'full_terminal.log'}")
    print(f"   - Supervisor:    {session_dir / 'supervisor.log'}")
    print(f"⚙️  Config: {session_dir / 'config.json'}\n")

    asyncio.run(main(problem_file, session_dir))
