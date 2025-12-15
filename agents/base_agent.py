"""
Base Agent Class - Abstract interface for all specialized agents
동기 방식으로 수정됨
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..memory import ContextMemory
from ..clients import LLMClient, WebSearchClient, get_client
from ..prompts import PromptManager


class BaseAgent(ABC):
    """Abstract base class for all specialized agents"""
    
    def __init__(
        self, 
        name: str, 
        memory: ContextMemory = None, 
        config: Dict[str, Any] = None, 
        llm_client: Optional[LLMClient] = None, 
        web_search: Optional[WebSearchClient] = None
    ):
        self.name = name
        self.memory = memory
        self.config = config or {}
        self.llm = llm_client or get_client()
        self.web_search = web_search
        self.prompt_manager = PromptManager()
        self.logger = logging.getLogger(f"Agent.{name}")
    
    @abstractmethod
    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary task (sync)"""
        pass
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message"""
        log_func = getattr(self.logger, level)
        log_func(f"[{self.name}] {message}")
