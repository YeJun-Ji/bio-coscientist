"""
BioCoScientist - Problem-Agnostic Bio Research AI System

New Architecture (v3.0):
- Dynamic research planning based on LLM analysis
- Adaptive task generation and worker management  
- No predefined problem types - handles any biomedical research question
- ConfigurationAgent + SupervisorAgent + WorkerPool orchestration

Usage:
    from biocoscientist import BioCoScientist
    
    scientist = BioCoScientist()
    results = await scientist.research("Your research goal...")
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
from .external_apis import LLMClient, get_client
from .biocoscientist import BioCoScientist

__version__ = "3.0.0"
__all__ = [
    "BioCoScientist",
    "LLMClient",
    "get_client"
]
