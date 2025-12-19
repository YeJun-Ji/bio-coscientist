"""
External APIs - Non-MCP external service clients
Includes OpenAI (LLM) and other third-party APIs
"""

from .openai_client import LLMClient, get_client

__all__ = [
    "LLMClient",
    "get_client",
]
