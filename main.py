"""
BioCoScientist - 통합 바이오 연구 AI 시스템
OpenRouter API 기반 비동기 방식 구현
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from .clients import LLMClient, get_client

logger = logging.getLogger(__name__)


class ProblemType(Enum):
    """지원하는 문제 유형"""
    GENE_SIMILARITY = "gene_similarity"
    RNA_STABILITY = "rna_stability"
    PROTEIN_BINDER = "protein_binder"
    TARGET_DISCOVERY = "target_discovery"
    DRUG_REPOSITIONING = "drug_repositioning"
    GENERAL_BIO = "general_bio"
    
    @classmethod
    def korean_name(cls, pt) -> str:
        """문제 유형의 한글 이름"""
        names = {
            cls.GENE_SIMILARITY: "유전자 유사성 분석",
            cls.RNA_STABILITY: "RNA 안정성 예측",
            cls.PROTEIN_BINDER: "단백질 바인더 설계",
            cls.TARGET_DISCOVERY: "치료 표적 발견",
            cls.DRUG_REPOSITIONING: "약물 재배치",
            cls.GENERAL_BIO: "일반 바이오 연구"
        }
        return names.get(pt, "바이오 연구")


class BioCoScientist:
    """
    통합 바이오 연구 AI 시스템
    
    5가지 문제 유형 지원:
    1. 유전자 기능 유사성 분석
    2. RNA 안정성 메커니즘 조사
    3. 단백질 바인더 설계
    4. 치료 표적 발견
    5. 약물 재배치
    
    5-에이전트 파이프라인:
    Generation → Reflection → Ranking → Evolution → Meta-Review
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        verbose: bool = True,
        output_dir: str = "reports"
    ):
        """
        BioCoScientist 초기화
        
        Args:
            llm_client: 커스텀 LLM 클라이언트 (None이면 기본 클라이언트 사용)
            verbose: 상세 출력 여부
            output_dir: 보고서 저장 디렉토리
        """
        self.llm = llm_client or get_client()
        self.verbose = verbose
        self.output_dir = output_dir
        self.results = {}
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("BioCoScientist 시스템 초기화 완료")
    
    def _log(self, message: str):
        """조건부 콘솔 출력"""
        if self.verbose:
            print(message)
    
    def detect_problem_type(self, problem: str) -> ProblemType:
        """문제 유형 자동 감지"""
        text = problem.lower()
        
        keywords = {
            ProblemType.GENE_SIMILARITY: [
                "유전자 유사성", "gene similarity", "gene function", "유전자 기능",
                "발현 상관", "expression correlation", "t cell gene", "cd69", "cd25"
            ],
            ProblemType.RNA_STABILITY: [
                "rna 안정성", "rna stability", "poly(a)", "cre", "nanopore",
                "mrna", "전사체", "drna-seq", "rna 구조"
            ],
            ProblemType.PROTEIN_BINDER: [
                "binder", "mini-binder", "단백질 결합", "protein binder",
                "tnfr", "binding affinity", "kd", "결합 친화도", "tnbc"
            ],
            ProblemType.TARGET_DISCOVERY: [
                "표적 발견", "target discovery", "therapeutic target", "치료 표적",
                "il-11", "fibrosis", "ppi network", "바이오마커"
            ],
            ProblemType.DRUG_REPOSITIONING: [
                "약물 재배치", "drug repositioning", "drug repurposing",
                "기존 약물", "새로운 적응증", "signature reversal"
            ]
        }
        
        scores = {pt: 0 for pt in ProblemType}
        for problem_type, kw_list in keywords.items():
            for kw in kw_list:
                if kw in text:
                    scores[problem_type] += 1
        
        max_score = max(scores.values())
        if max_score > 0:
            for pt, score in scores.items():
                if score == max_score:
                    return pt
        
        return ProblemType.GENERAL_BIO
    
    def _get_problem_context(self, problem_type: ProblemType) -> str:
        """문제 유형별 추가 컨텍스트"""
        contexts = {
            ProblemType.GENE_SIMILARITY: """
### 유전자 유사성 분석 컨텍스트:
- 서열 유사성과 구조적 특성 분석
- 기능적 annotation 비교
- PPI 네트워크에서의 역할
- 발현 패턴 분석
- 진화적 관계 및 보존성
""",
            ProblemType.RNA_STABILITY: """
### RNA 안정성 분석 컨텍스트:
- RNA 2차 구조 예측
- 열역학적 안정성 계산
- 구조적 모티프 식별
- 환경 조건에 따른 안정성 변화
- 실험적 검증 방법
""",
            ProblemType.PROTEIN_BINDER: """
### 단백질 바인더 설계 컨텍스트:
- 결합 부위 분석 및 선정
- 생물물리적 제약 조건 (길이, 전하, 안정성)
- AI 기반 서열 설계 (AlphaFold, RosettaFold, ProteinMPNN)
- 결합 특성 예측 (KD, kon, koff)
- 오프타깃 스크리닝
- 면역원성 평가
""",
            ProblemType.TARGET_DISCOVERY: """
### 치료 표적 발견 컨텍스트:
- 오믹스 데이터 통합 분석
- 네트워크 기반 표적 예측
- 드러거빌리티 평가
- 기능적 검증 실험 설계
- 바이오마커 개발
""",
            ProblemType.DRUG_REPOSITIONING: """
### 약물 재배치 컨텍스트:
- 약물-질병 네트워크 분석
- 전사체 시그니처 매칭
- 구조 기반 표적 예측
- 임상 데이터 마이닝
- AI/ML 기반 예측 모델
""",
            ProblemType.GENERAL_BIO: """
### 일반 바이오 연구 컨텍스트:
- 과학적 방법론 적용
- 가설 기반 연구 설계
- 실험적 검증 전략
- 데이터 분석 및 해석
"""
        }
        return contexts.get(problem_type, contexts[ProblemType.GENERAL_BIO])
    
    # ===== 5 Agent Pipeline =====
    
    async def run_generation_agent(self, problem: str, problem_type: ProblemType) -> str:
        """Generation Agent: 가설 생성"""
        self._log("\n" + "="*60)
        self._log("🧬 STEP 1: Generation Agent - 가설 생성")
        self._log("="*60)
        
        context = self._get_problem_context(problem_type)
        
        response = await self.llm.generate(
            messages=[{
                "role": "user",
                "content": f"""다음 연구 문제에 대해 3-5개의 창의적이고 실현 가능한 가설을 생성해주세요.

문제:
{problem}

각 가설에 대해 다음 형식으로 제시해주세요:
## 가설 [번호]: [제목]
**핵심 주장**: ...
**과학적 근거**: ...
**제안하는 접근법**: ...
**예상 결과**: ...
**검증 방법**: ..."""
            }],
            system=f"""당신은 BioCoScientist의 Generation Agent입니다.
주어진 연구 문제에 대해 혁신적이고 실현 가능한 가설들을 생성합니다.
{context}"""
        )
        
        self._log(response)
        return response
    
    async def run_reflection_agent(self, hypotheses: str, problem_type: ProblemType) -> str:
        """Reflection Agent: 가설 검토"""
        self._log("\n" + "="*60)
        self._log("🔍 STEP 2: Reflection Agent - 가설 검토 및 평가")
        self._log("="*60)
        
        response = await self.llm.generate(
            messages=[{
                "role": "user",
                "content": f"""다음 가설들을 검토하고 각각에 대해 상세한 피드백을 제공해주세요.

{hypotheses}

각 가설에 대해 평가해주세요:
1. **강점** (3개 이상)
2. **약점** (3개 이상)
3. **과학적 타당성** (1-10점)
4. **실현 가능성** (1-10점)
5. **혁신성** (1-10점)
6. **개선 제안** (구체적인 방안)"""
            }],
            system="""당신은 BioCoScientist의 Reflection Agent입니다.
생성된 가설들을 다각도로 검토하고 피드백을 제공합니다.

검토 기준:
1. 과학적 타당성 (Scientific validity)
2. 기술적 실현 가능성 (Technical feasibility)
3. 혁신성 (Novelty)
4. 임상적 의미 (Clinical relevance)
5. 리스크/보상 비율 (Risk/reward ratio)"""
        )
        
        self._log(response)
        return response
    
    async def run_ranking_agent(self, hypotheses: str, reviews: str) -> str:
        """Ranking Agent: 가설 순위화"""
        self._log("\n" + "="*60)
        self._log("📊 STEP 3: Ranking Agent - 가설 순위화")
        self._log("="*60)
        
        response = await self.llm.generate(
            messages=[{
                "role": "user",
                "content": f"""다음 가설들과 검토 결과를 바탕으로 최종 순위를 결정해주세요.

가설들:
{hypotheses[:3000]}

검토 결과:
{reviews[:3000]}

다음 형식으로 순위를 제시해주세요:
## 최종 순위
| 순위 | 가설 | 종합 점수 | 핵심 강점 |
|------|------|-----------|-----------|

## 1위 가설 상세 분석
...

## 권장 연구 전략
..."""
            }],
            system="""당신은 BioCoScientist의 Ranking Agent입니다.
Tournament-style 방식으로 가설들을 비교하고 최종 순위를 결정합니다.

평가 가중치:
- 과학적 타당성: 30%
- 실현 가능성: 25%
- 혁신성: 20%
- 임상적 의미: 15%
- 리스크/보상: 10%"""
        )
        
        self._log(response)
        return response
    
    async def run_evolution_agent(self, ranking: str, problem: str) -> str:
        """Evolution Agent: 가설 진화"""
        self._log("\n" + "="*60)
        self._log("🔬 STEP 4: Evolution Agent - 가설 진화 및 구체화")
        self._log("="*60)
        
        response = await self.llm.generate(
            messages=[{
                "role": "user",
                "content": f"""원래 문제:
{problem[:1500]}

순위화 결과:
{ranking[:2500]}

다음을 수행해주세요:
1. 상위 가설의 강점을 더욱 강화
2. 약점을 보완하는 구체적 방안 제시
3. 실험 계획 구체화
4. 예상되는 도전과 대응 전략
5. 최종 연구 로드맵 제시"""
            }],
            system="""당신은 BioCoScientist의 Evolution Agent입니다.
상위 가설을 더욱 발전시키고 구체화합니다.

진화 방법:
1. Grounding: 실험적 근거 강화
2. Coherence: 논리적 일관성 향상
3. Combination: 여러 접근법 통합
4. Simplification: 핵심 요소 도출
5. Divergent: 새로운 방향 탐색""",
            max_tokens=8192
        )
        
        self._log(response)
        return response
    
    async def run_meta_review(self, all_results: Dict[str, str], problem: str, problem_type: ProblemType) -> str:
        """Meta-Review Agent: 종합 분석 및 최종 보고서"""
        self._log("\n" + "="*60)
        self._log("📋 STEP 5: Meta-Review Agent - 종합 연구 보고서 생성")
        self._log("="*60)
        
        problem_type_korean = ProblemType.korean_name(problem_type)
        
        summary = f"""
### 원래 문제:
{problem[:1000]}

### Generation 결과 요약:
{all_results.get('generation', '')[:1500]}

### Reflection 결과 요약:
{all_results.get('reflection', '')[:1500]}

### Ranking 결과 요약:
{all_results.get('ranking', '')[:1500]}

### Evolution 결과 요약:
{all_results.get('evolution', '')[:1500]}
"""
        
        response = await self.llm.generate(
            messages=[{
                "role": "user",
                "content": f"""전체 연구 과정을 종합하여 최종 보고서를 작성해주세요.

{summary}

다음 구조로 최종 보고서를 작성해주세요:

# {problem_type_korean} 연구 종합 보고서

## 1. 연구 개요
(배경, 목적, 문제 정의)

## 2. 핵심 가설 요약
(생성된 주요 가설들과 평가 결과)

## 3. 제안된 연구 파이프라인
(단계별 방법론, 도구, 입출력 관계)

## 4. 예상 결과 및 성과 지표
(정량적/정성적 예상 결과)

## 5. 임상적/학술적 의의
(연구의 중요성과 기대 효과)

## 6. 한계점 및 향후 과제
(도전 과제와 해결 방안)

## 7. 결론
(핵심 메시지 요약)"""
            }],
            system=f"""당신은 BioCoScientist의 Meta-Review Agent입니다.
전체 연구 과정을 종합하여 최종 보고서를 작성합니다.
문제 유형: {problem_type_korean}""",
            max_tokens=8192
        )
        
        self._log(response)
        return response
    
    # ===== Main Entry Point =====
    
    async def run(self, problem: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        전체 파이프라인 실행 (비동기)
        
        Args:
            problem: 연구 문제 설명
            output_file: 출력 파일 경로 (None이면 자동 생성)
        
        Returns:
            연구 결과 딕셔너리
        """
        start_time = datetime.now()
        
        # 문제 유형 감지
        problem_type = self.detect_problem_type(problem)
        problem_type_korean = ProblemType.korean_name(problem_type)
        
        self._log("\n" + "="*70)
        self._log("🚀 BioCoScientist 연구 시작")
        self._log("="*70)
        self._log(f"📋 감지된 문제 유형: {problem_type_korean}")
        self._log(f"⏰ 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            'problem': problem,
            'problem_type': problem_type.value,
            'problem_type_korean': problem_type_korean,
            'start_time': start_time.isoformat()
        }
        
        try:
            # 5-Agent Pipeline (비동기)
            self._log("\n⏳ Generation Agent 실행 중...")
            results['generation'] = await self.run_generation_agent(problem, problem_type)
            
            self._log("\n⏳ Reflection Agent 실행 중...")
            results['reflection'] = await self.run_reflection_agent(results['generation'], problem_type)
            
            self._log("\n⏳ Ranking Agent 실행 중...")
            results['ranking'] = await self.run_ranking_agent(results['generation'], results['reflection'])
            
            self._log("\n⏳ Evolution Agent 실행 중...")
            results['evolution'] = await self.run_evolution_agent(results['ranking'], problem)
            
            self._log("\n⏳ Meta-Review Agent 실행 중...")
            results['meta_review'] = await self.run_meta_review(results, problem, problem_type)
            
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            self._log(f"\n❌ 오류 발생: {e}")
            logger.exception("Pipeline error")
        
        # 완료 시간
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['duration'] = str(end_time - start_time)
        
        # 보고서 저장
        if results['status'] == 'success':
            self._save_report(results, output_file)
        
        self._log("\n" + "="*70)
        self._log("🎉 BioCoScientist 연구 완료!")
        self._log(f"⏱️ 소요 시간: {results['duration']}")
        self._log("="*70)
        
        return results
    
    def run_sync(self, problem: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """동기 방식 실행 (편의 메서드)"""
        return asyncio.run(self.run(problem, output_file))
    
    def _save_report(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """보고서 파일 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        problem_type = results.get('problem_type', 'research')
        
        # 파일명 생성
        if output_file is None:
            txt_file = os.path.join(self.output_dir, f"BioCoScientist_Report_{problem_type}_{timestamp}.txt")
            json_file = os.path.join(self.output_dir, f"BioCoScientist_Results_{problem_type}_{timestamp}.json")
        else:
            base = os.path.splitext(output_file)[0]
            txt_file = f"{base}.txt"
            json_file = f"{base}.json"
        
        # 텍스트 보고서 저장
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BioCoScientist 연구 보고서\n")
            f.write("="*70 + "\n\n")
            f.write(f"문제 유형: {results.get('problem_type_korean', 'N/A')}\n")
            f.write(f"생성 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}\n")
            f.write(f"소요 시간: {results.get('duration', 'N/A')}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("원래 문제\n")
            f.write("="*70 + "\n\n")
            f.write(results.get('problem', '') + "\n")
            
            sections = [
                ("1. 가설 생성 (Generation Agent)", results.get('generation', '')),
                ("2. 가설 검토 (Reflection Agent)", results.get('reflection', '')),
                ("3. 가설 순위화 (Ranking Agent)", results.get('ranking', '')),
                ("4. 가설 진화 (Evolution Agent)", results.get('evolution', '')),
                ("5. 종합 보고서 (Meta-Review Agent)", results.get('meta_review', ''))
            ]
            
            for title, content in sections:
                f.write(f"\n{'='*70}\n")
                f.write(f"{title}\n")
                f.write(f"{'='*70}\n\n")
                f.write(content + "\n")
        
        self._log(f"\n📁 보고서 저장됨: {txt_file}")
        
        # JSON 결과 저장
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self._log(f"📁 JSON 결과 저장됨: {json_file}")
