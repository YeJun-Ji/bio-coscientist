"""
Base Agent Class - Abstract interface for all specialized agents
Supports both sync and async execution with MCP tool access
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..memory import ContextMemory
from ..external_apis import LLMClient, get_client
from ..prompts import PromptManager


class BaseAgent(ABC):
    """Abstract base class for all specialized agents"""

    def __init__(
        self,
        name: str = None,
        memory: ContextMemory = None,
        config: Dict[str, Any] = None,
        llm_client: Optional[LLMClient] = None,
        tool_registry=None,
        mcp_server_manager=None,
        worker_id: Optional[str] = None,
        **kwargs
    ):
        # Allow name to be set by subclass
        self.name = name or self.__class__.__name__.replace("Agent", "").lower()
        self.worker_id = worker_id
        self.memory = memory
        self.config = config or {}
        self.llm = llm_client or get_client()
        self.prompt_manager = PromptManager()

        # Create unique logger per worker to prevent log mixing
        # Format: "Agent.{worker_id}" for worker-bound agents
        # Format: "Agent.{name}" for standalone agents (no worker)
        if worker_id:
            self.logger = logging.getLogger(f"Agent.{worker_id}")
        else:
            self.logger = logging.getLogger(f"Agent.{self.name}")

        # Enable propagation to root logger so full_terminal.log receives all logs
        # Only disable in multi-worker scenarios where separate log files are needed
        self.logger.propagate = True

        # MCP infrastructure (optional - not all agents need it)
        self.tool_registry = tool_registry
        self.mcp_manager = mcp_server_manager

    async def call_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Call a single MCP tool

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result or None if MCP not available
        """
        if not self.mcp_manager:
            self.logger.warning(f"MCP manager not available, skipping {tool_name}")
            return None

        try:
            result = await self.mcp_manager.call_tool(tool_name, arguments)
            return {"name": tool_name, "result": result, "status": "success"}
        except Exception as e:
            self.logger.error(f"MCP tool {tool_name} failed: {e}")
            return {"name": tool_name, "result": None, "error": str(e), "status": "error"}

    async def call_mcp_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Call multiple MCP tools in parallel using asyncio.gather

        Args:
            tool_calls: List of dicts with 'name' and 'arguments' keys

        Returns:
            List of results from all tool calls

        Example:
            results = await self.call_mcp_tools_parallel([
                {"name": "search_proteins", "arguments": {"query": "p53"}},
                {"name": "search_genes", "arguments": {"query": "TP53"}}
            ])
        """
        if not self.mcp_manager:
            self.logger.warning("MCP manager not available, skipping parallel tool calls")
            return []

        if not tool_calls:
            return []

        # Create coroutines for all tool calls
        coroutines = [
            self.call_mcp_tool(tc["name"], tc.get("arguments", {}))
            for tc in tool_calls
        ]

        # Execute all in parallel
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results - handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = tool_calls[i]["name"]
                self.logger.error(f"Parallel tool {tool_name} raised exception: {result}")
                processed_results.append({
                    "name": tool_name,
                    "result": None,
                    "error": str(result),
                    "status": "error"
                })
            else:
                processed_results.append(result)

        return processed_results

    def has_mcp_access(self) -> bool:
        """Check if this agent has MCP tool access"""
        return self.mcp_manager is not None
    
    @abstractmethod
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task (async).
        
        Args:
            task: Task parameters (varies by agent type)
        
        Returns:
            Task results
        """
        pass
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with agent name prefix"""
        log_func = getattr(self.logger, level)
        # Include agent name in message for clarity
        log_func(f"[{self.name}] {message}")
