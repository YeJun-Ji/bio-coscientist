"""
BioCoScientist - 통합 바이오 연구 AI 시스템

5가지 문제 유형 지원:
1. 유전자 기능 유사성 분석 (Gene Similarity)
2. RNA 안정성 메커니즘 조사 (RNA Stability)
3. 단백질 바인더 설계 (Protein Binder)
4. 치료 표적 발견 (Target Discovery)
5. 약물 재배치 (Drug Repositioning)

사용법:
    from biocoscientist import BioCoScientist
    
    scientist = BioCoScientist()
    results = scientist.run("연구 문제 설명...")
"""

# .env 파일 자동 로드
from dotenv import load_dotenv
import os

_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()

# 핵심 클래스 export
from .main import BioCoScientist, ProblemType
from .clients import LLMClient, get_client

__version__ = "2.0.0"
__all__ = [
    "BioCoScientist",
    "ProblemType",
    "LLMClient",
    "get_client"
]
