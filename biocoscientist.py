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
def setup_logging():
    """Set up logging to both file and console"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"biocoscientist_{timestamp}.log"
    
    # Root logger 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 핸들러 추가
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # httpx 로거는 WARNING 레벨로 설정 (HTTP 요청 로그 제거)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return log_file

# Support both relative imports (when used as module) and absolute imports (when run directly)
try:
    from .core import ResearchGoal, Hypothesis, ProblemType
    from .agents import SupervisorAgent
    from .problems import (
        GeneSimilarityHandler,
        RNAStabilityHandler,
        ProteinBinderHandler,
        TargetDiscoveryHandler,
        DrugRepositioningHandler
    )
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from biocoscientist.core import ResearchGoal, Hypothesis, ProblemType
    from biocoscientist.agents import SupervisorAgent
    from biocoscientist.problems import (
        GeneSimilarityHandler,
        RNAStabilityHandler,
        ProteinBinderHandler,
        TargetDiscoveryHandler,
        DrugRepositioningHandler
    )

logger = logging.getLogger(__name__)


class BioCoScientist:
    """
    Bio AI Co-Scientist - Multi-problem biomedical research assistant
    
    Supports 5 problem types:
    1. Gene Function Similarity Analysis
    2. RNA Stability Mechanism Investigation
    3. Protein Binder Design
    4. Therapeutic Target Discovery
    5. Drug Repositioning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Bio AI Co-Scientist system"""
        self.config = config or self.default_config()
        self.supervisor = SupervisorAgent(self.config)
        self.logger = logging.getLogger("BioCoScientist")
        self.logger.info("="*80)
        self.logger.info("BioCoScientist System Initialized")
        self.logger.info("="*80)
        
        # Initialize problem-specific handlers
        self.problem_handlers = {
            ProblemType.GENE_SIMILARITY: GeneSimilarityHandler(),
            ProblemType.RNA_STABILITY: RNAStabilityHandler(),
            ProblemType.PROTEIN_BINDER: ProteinBinderHandler(),
            ProblemType.TARGET_DISCOVERY: TargetDiscoveryHandler(),
            ProblemType.DRUG_REPOSITIONING: DrugRepositioningHandler()
        }
        
        self.logger.info("Bio AI Co-Scientist system initialized")
    
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
                "techniques": ["literature", "debate", "assumptions", "expansion"]
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
    
    async def research(
        self,
        goal_description: str,
        domain: str,
        focus_areas: List[str],
        problem_type: Optional[ProblemType] = None,
        max_iterations: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Conduct automated research for a given goal.
        
        Args:
            goal_description: Description of the research goal
            domain: Research domain (e.g., "biology", "bioinformatics")
            focus_areas: List of specific focus areas
            problem_type: Specific problem type (auto-detected if None)
            max_iterations: Maximum number of research iterations
            **kwargs: Additional configuration options
        
        Returns:
            Research results including top hypotheses and overview
        """
        self.logger.info(f"Starting research: {goal_description}")
        
        # Auto-detect problem type if not specified
        if problem_type is None:
            problem_type = self._detect_problem_type(goal_description, domain, focus_areas)
            self.logger.info(f"Auto-detected problem type: {problem_type.value}")
        
        # Create research goal
        research_goal = ResearchGoal(
            goal_id=f"goal_{datetime.now().timestamp()}",
            description=goal_description,
            domain=domain,
            focus_areas=focus_areas,
            constraints=kwargs.get("constraints", {}),
            success_criteria=kwargs.get("success_criteria", []),
            created_at=datetime.now(),
            problem_type=problem_type
        )
        
        # Get problem-specific handler
        handler = self.problem_handlers.get(problem_type)
        
        # Add problem-specific context to config
        if handler:
            self.config["problem_handler"] = handler
            self.config["problem_type"] = problem_type
            self.logger.info(f"Using handler: {handler.problem_type}")
        
        # Run research cycle
        results = await self.supervisor.run_research_cycle(
            research_goal=research_goal,
            max_iterations=max_iterations,
            hypotheses_per_iteration=kwargs.get("hypotheses_per_iteration", 10)
        )
        
        # Add problem-specific validation
        if handler:
            results["expected_outputs"] = handler.get_expected_outputs(research_goal)
            results["domain_knowledge"] = handler.get_domain_knowledge()
        
        # problem_type을 인스턴스에 저장 (보고서 생성 시 사용)
        self.problem_type = problem_type
        
        self.logger.info("Research complete")
        
        return results
    
    def _detect_problem_type(
        self,
        description: str,
        domain: str,
        focus_areas: List[str]
    ) -> ProblemType:
        """
        Auto-detect problem type from description and focus areas
        """
        text = f"{description} {domain} {' '.join(focus_areas)}".lower()
        
        # Gene similarity indicators
        if any(term in text for term in ["gene similarity", "gene function", "expression correlation", "t cell gene"]):
            return ProblemType.GENE_SIMILARITY
        
        # RNA stability indicators  
        if any(term in text for term in ["rna stability", "poly(a)", "cre", "nanopore", "drna-seq", "mrna stabilization"]):
            return ProblemType.RNA_STABILITY
        
        # Protein binder indicators
        if any(term in text for term in ["binder design", "mini-binder", "protein binder", "tnfr", "binding affinity", "kd <"]):
            return ProblemType.PROTEIN_BINDER
        
        # Target discovery indicators
        if any(term in text for term in ["target discovery", "il-11", "fibrosis", "ppi network", "therapeutic target"]):
            return ProblemType.TARGET_DISCOVERY
        
        # Drug repositioning indicators
        if any(term in text for term in ["drug repositioning", "exhaustion", "signature reversal", "drug repurposing"]):
            return ProblemType.DRUG_REPOSITIONING
        
        # Default to gene similarity if unclear
        self.logger.warning("Could not auto-detect problem type, defaulting to GENE_SIMILARITY")
        return ProblemType.GENE_SIMILARITY
    
    def get_research_overview(self) -> Dict[str, Any]:
        """Get the latest research overview"""
        overview = self.supervisor.memory.research_overviews[-1] if self.supervisor.memory.research_overviews else None
        return overview
    
    def get_top_hypotheses(self, n: int = 10) -> List[Hypothesis]:
        """Get top N hypotheses by ranking"""
        return self.supervisor.memory.get_top_hypotheses(n)
    
    def get_problem_handler(self, problem_type: ProblemType):
        """Get handler for specific problem type"""
        return self.problem_handlers.get(problem_type)
    
    def list_problem_types(self) -> List[str]:
        """List all supported problem types"""
        return [pt.value for pt in ProblemType]
    
    async def parse_problem_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a problem description file and extract research parameters dynamically.
        
        Args:
            file_path: Path to the problem description file
        
        Returns:
            Dictionary with goal_description, domain, focus_areas, constraints, success_criteria
        """
        print(f"[DEBUG] 1. Starting parse_problem_file: {file_path}")
        self.logger.info(f"Parsing problem file: {file_path}")
        
        # Read file content
        print(f"[DEBUG] 2. Reading file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            problem_text = f.read()
        print(f"[DEBUG] 3. File read complete, length: {len(problem_text)}")
        
        # Use LLM to extract structured information
        print(f"[DEBUG] 4. Importing LLMClient...")
        try:
            from .clients import LLMClient
        except ImportError:
            from biocoscientist.clients import LLMClient
        
        print(f"[DEBUG] 5. Creating LLMClient...")
        llm_config = self.config.get("llm", {})
        print(f"[DEBUG] 6. LLM config provider: {llm_config.get('provider')}, model: {llm_config.get('model')}")
        
        # API 키 가져오기 - config에서 이미 환경변수 읽어서 저장됨
        api_key = llm_config.get("api_key")
        if not api_key:
            print(f"[DEBUG] ❌ No API key found - please set OPENROUTER_API_KEY in .env file")
            print(f"[DEBUG] Using fallback parsing without LLM")
            return {
                "goal_description": problem_text[:500],
                "domain": "Biomedical Research",
                "focus_areas": ["Analysis", "Research"],
                "constraints": {},
                "success_criteria": ["Complete analysis"]
            }
        
        print(f"[DEBUG] ✅ API key found: {api_key[:20]}...")
        
        try:
            llm_client = LLMClient(
                provider=llm_config.get("provider"),
                model=llm_config.get("model"),
                api_key=api_key,
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 8192)
            )
            print(f"[DEBUG] 7. LLMClient created successfully")
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
- success_criteria는 평가 기준이나 달성 목표
"""
        
        print(f"[DEBUG] 8. Starting LLM API call...")
        try:
            result = await llm_client.generate_json(
                messages=[{"role": "user", "content": extraction_prompt}],
                system="You are an expert in biomedical research problem analysis. Extract structured information accurately."
            )
            
            print(f"[DEBUG] 9. LLM API call successful")
            self.logger.info("Successfully parsed problem file")
            self.logger.debug(f"Extracted: {result}")
            
            return result
            
        except Exception as e:
            print(f"[DEBUG] 10. LLM API call failed: {e}")
            self.logger.error(f"Failed to parse problem file: {e}")
            # Fallback to basic extraction
            print(f"[DEBUG] 11. Using fallback parsing")
            return {
                "goal_description": problem_text[:500],
                "domain": "Biomedical Research",
                "focus_areas": ["Analysis", "Research"],
                "constraints": {},
                "success_criteria": ["Complete analysis"]
            }
    
    def export_results(self, output_path: str) -> None:
        """Export research results to file"""
        import json
        from pathlib import Path
        
        results = {
            "hypotheses": [h.__dict__ for h in self.supervisor.memory.hypotheses.values()],
            "reviews": [r.__dict__ for r in self.supervisor.memory.reviews.values()],
            "overviews": self.supervisor.memory.research_overviews
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results exported to {output_path}")
        
        # 보고서 자동 생성
        try:
            from biocoscientist.utils.report_generator import generate_report_from_json
            import re
            
            # reports 폴더 경로 생성
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            # 로그 파일명에서 타임스탬프 추출 (biocoscientist_YYYYMMDD_HHMMSS.log)
            log_files = sorted(Path("logs").glob("biocoscientist_*.log"), reverse=True)
            if log_files:
                log_name = log_files[0].stem  # biocoscientist_20251215_055156
                timestamp = log_name.replace("biocoscientist_", "")  # 20251215_055156
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # problem_type 가져오기
            problem_type = getattr(self, 'problem_type', 'research')
            if hasattr(problem_type, 'value'):
                problem_type = problem_type.value
            
            # 파일명 형식: BioCoScientist_Report_{problem_type}_{timestamp}.txt
            report_name = f"BioCoScientist_Report_{problem_type}_{timestamp}"
            
            # Full 보고서 생성
            full_report_path = str(reports_dir / f"{report_name}.txt")
            generate_report_from_json(output_path, full_report_path, "full")
            self.logger.info(f"Full report generated: {full_report_path}")
            
            # Summary 보고서 생성
            summary_report_path = str(reports_dir / f"{report_name}_summary.txt")
            generate_report_from_json(output_path, summary_report_path, "summary")
            self.logger.info(f"Summary report generated: {summary_report_path}")
            
            print(f"\n📊 Reports Generated:")
            print(f"  - Full Report: {full_report_path}")
            print(f"  - Summary Report: {summary_report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
            print(f"⚠️  Report generation failed: {e}")


# ============================================================================
# Example Usage
# ============================================================================

async def main(problem_file_path: str = None):
    """
    Example usage of the Bio AI Co-Scientist system with dynamic problem file parsing.
    
    Args:
        problem_file_path: Path to problem description file (e.g., 'problems/tnbc_minibinder.txt')
                          If None, uses default static example
    """
    
    # Initialize the system
    bio_coscientist = BioCoScientist()
    
    # Parse problem file if provided, otherwise use static example
    if problem_file_path:
        print(f"\n📄 Parsing problem file: {problem_file_path}")
        problem_params = await bio_coscientist.parse_problem_file(problem_file_path)
        
        goal_description = problem_params["goal_description"]
        domain = problem_params["domain"]
        focus_areas = problem_params["focus_areas"]
        constraints = problem_params.get("constraints", {})
        success_criteria = problem_params.get("success_criteria", [])
        
        print("\n✅ Extracted Parameters:")
        print(f"  Goal: {goal_description[:100]}...")
        print(f"  Domain: {domain}")
        print(f"  Focus Areas: {focus_areas}")
        print(f"  Constraints: {constraints}")
        print(f"  Success Criteria: {success_criteria}")
    else:
        # Fallback to static example
        print("\n📝 Using static example (Protein Binder Design)")
        goal_description = """
        Design AI-based mini-binder therapeutics targeting TNFR1/2 for Triple-Negative 
        Breast Cancer (TNBC) treatment by modulating TNFα-ΔNp63α signaling axis.
        """
        
        domain = "Protein Engineering & Drug Discovery"
        focus_areas = [
            "AI-based protein binder design",
            "TNFR1/2 selective targeting",
            "Binding affinity optimization (KD, kon, koff)",
            "Off-target receptor screening"
        ]
        constraints = {
            "binder_type": "mini-binder",
            "target_receptors": "TNFR1 and/or TNFR2",
            "affinity_target": "KD < 10 nM"
        }
        success_criteria = [
            "High TNFR binding specificity",
            "Minimal off-target effects",
            "Low immunogenicity"
        ]
    
    # Run research (problem type auto-detected)
    print("\n🚀 Starting research...\n")
    results = await bio_coscientist.research(
        goal_description=goal_description,
        domain=domain,
        focus_areas=focus_areas,
        max_iterations=2,
        hypotheses_per_iteration=5,
        constraints=constraints,
        success_criteria=success_criteria
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESEARCH COMPLETE")
    print("="*80)
    print(f"\nProblem Type: {results.get('problem_type', 'N/A')}")
    print(f"Total hypotheses generated: {results['total_hypotheses']}")
    print(f"Iterations completed: {results['iterations_completed']}")
    
    print("\n--- Top 5 Hypotheses ---")
    for i, hyp in enumerate(results['top_hypotheses'][:5], 1):
        print(f"\n{i}. [{hyp.id}]")
        print(f"   Summary: {hyp.summary}")
        print(f"   Elo Rating: {hyp.elo_rating:.1f}")
        print(f"   Status: {hyp.status.value}")
    
    # Export results
    bio_coscientist.export_results("research_results.json")


if __name__ == "__main__":
    import asyncio
    import sys
    
    # 로깅 설정
    log_file = setup_logging()
    
    # Get problem file path from command line argument
    problem_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if problem_file:
        print(f"\n{'='*80}")
        print(f"🔬 BioCoScientist - Dynamic Problem Solving")
        print(f"{'='*80}")
        print(f"Problem File: {problem_file}")
        print(f"📋 Log File: {log_file}")
    else:
        print(f"\n{'='*80}")
        print(f"🔬 BioCoScientist - Static Example Mode")
        print(f"{'='*80}")
        print("Usage: python biocoscientist.py <problem_file.txt>")
        print("Running with default static example...")
        print(f"📋 Log File: {log_file}")
    
    asyncio.run(main(problem_file))
