"""
Tool Registry - Central management for all tools (Dynamic MCP Loading)

This module provides dynamic tool registration from MCP servers.
Tools are discovered at runtime from connected MCP servers instead of
being manually defined in code.

Architecture:
1. MCPServerManager starts MCP servers and lists their tools
2. ToolRegistry loads tools dynamically from ServerManager
3. LLM receives tool definitions for function calling
4. ToolExecutor routes tool calls to appropriate MCP server
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """
    Tool definition with metadata

    For MCP tools, this is dynamically created from MCP server's tool schema.
    For legacy tools (websearch, etc.), this can be manually created.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    server: Optional[str] = None  # MCP server name (e.g., "kegg", "rosetta")
    problem_types: List[str] = None  # Which problems can use this tool
    stages: List[str] = None   # Which agent stages can use this tool
    category: str = "collection"  # "collection" | "analysis" - for 2-Chain architecture
    cost: str = "low"         # "low", "medium", "high"
    priority: int = 1         # Higher = more important
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.problem_types is None:
            self.problem_types = ["all"]
        if self.stages is None:
            self.stages = ["all"]
        if self.metadata is None:
            self.metadata = {}
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """
    Central registry for all tools with dynamic MCP loading
    
    All tools are loaded dynamically from MCP Servers (KEGG, Rosetta, etc.)
    """
    
    def __init__(self, mcp_server_manager=None):
        """
        Initialize Tool Registry
        
        Args:
            mcp_server_manager: MCPServerManager instance for dynamic tool loading
        """
        self._tools: Dict[str, ToolDefinition] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mcp_manager = mcp_server_manager
        
        # Load MCP tools dynamically if manager is provided
        if self.mcp_manager and self.mcp_manager.initialized:
            self._load_mcp_tools()
    
    def _load_mcp_tools(self):
        """Load tools dynamically from MCP servers"""
        if not self.mcp_manager:
            return
        
        mcp_tools = self.mcp_manager.get_all_tools()
        
        for mcp_tool in mcp_tools:
            # Extract tool information from MCP format
            func = mcp_tool.get("function", {})
            config = mcp_tool.get("_config")
            server_name = mcp_tool.get("_server")
            
            # Create ToolDefinition
            tool = ToolDefinition(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=func.get("parameters", {}),
                server=server_name,
                problem_types=config.problem_types if config else ["all"],
                stages=config.stages if config else ["all"],
                category=config.category if config else "collection",
                cost="low",  # Default cost
                priority=5,  # MCP tools have higher priority
                metadata={
                    "mcp_server": server_name,
                    "server_description": config.description if config else "",
                }
            )
            
            self.register_tool(tool)
        
        self.logger.info(f"Loaded {len(mcp_tools)} tools from MCP servers")
    
    def reload_mcp_tools(self):
        """Reload tools from MCP servers (useful after server changes)"""
        # Remove existing MCP tools
        self._tools = {
            name: tool for name, tool in self._tools.items()
            if tool.server is None  # Keep only legacy tools
        }
        
        # Reload MCP tools
        self._load_mcp_tools()
        
        self.logger.info(f"Reloaded MCP tools. Total: {len(self._tools)} tools")
    
    def register_tool(self, tool: ToolDefinition):
        """Register a tool definition"""
        if tool.name in self._tools:
            # MCP tools override legacy tools with same name
            existing = self._tools[tool.name]
            if tool.server and not existing.server:
                self.logger.info(
                    f"MCP tool '{tool.name}' replacing legacy tool"
                )
        
        self._tools[tool.name] = tool
        self.logger.debug(f"Registered tool: {tool.name} (server: {tool.server})")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a single tool by name"""
        return self._tools.get(name)
    
    def get_tools_for_problem(
        self,
        problem_type: str,
        stage: Optional[str] = None
    ) -> List[ToolDefinition]:
        """
        Get tools filtered by problem type and stage
        
        Args:
            problem_type: Problem type (e.g., "protein_binder_design")
            stage: Optional stage filter (e.g., "generation", "reflection")
        
        Returns:
            List of tool definitions matching criteria
        """
        tools = [
            tool for tool in self._tools.values()
            if ("all" in tool.problem_types or problem_type in tool.problem_types)
        ]
        
        if stage:
            tools = [
                tool for tool in tools
                if ("all" in tool.stages or stage in tool.stages)
            ]
        
        # Sort by priority (descending)
        tools.sort(key=lambda t: t.priority, reverse=True)
        return tools
    
    def get_tools_openai_format(
        self,
        problem_type: str,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI function calling format
        
        Args:
            problem_type: Problem type (e.g., "protein_binder_design")
            stage: Optional stage filter (e.g., "generation")
        
        Returns:
            List of tool definitions in OpenAI format for LLM function calling
        """
        tools = self.get_tools_for_problem(problem_type, stage)
        return [tool.to_openai_format() for tool in tools]

    # ========================================================================
    # Category-based Filtering (for 2-Chain Architecture)
    # ========================================================================

    def get_tools_by_category(
        self,
        category: str,
        problem_type: str = "all",
        stage: Optional[str] = None
    ) -> List[ToolDefinition]:
        """
        Get tools filtered by category (collection/analysis)

        Args:
            category: "collection" or "analysis"
            problem_type: Problem type filter (default: "all")
            stage: Optional stage filter

        Returns:
            List of tool definitions matching category and filters
        """
        tools = [
            tool for tool in self._tools.values()
            if tool.category == category
            and ("all" in tool.problem_types or problem_type in tool.problem_types)
        ]

        if stage:
            tools = [
                tool for tool in tools
                if ("all" in tool.stages or stage in tool.stages)
            ]

        tools.sort(key=lambda t: t.priority, reverse=True)
        return tools

    def get_collection_tools(
        self,
        problem_type: str = "all",
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get collection tools in OpenAI format (Chain 1: Data Collection)

        Collection tools gather data from databases:
        - kegg, uniprot, ncbi, scholar, reactome
        """
        tools = self.get_tools_by_category("collection", problem_type, stage)
        return [tool.to_openai_format() for tool in tools]

    def get_analysis_tools(
        self,
        problem_type: str = "all",
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get analysis tools in OpenAI format (Chain 2: Data Analysis)

        Analysis tools perform computational analysis:
        - rosetta, pymol, esmfold, blast
        """
        tools = self.get_tools_by_category("analysis", problem_type, stage)
        return [tool.to_openai_format() for tool in tools]

    def get_tool_stats_by_category(self) -> Dict[str, int]:
        """Get count of tools by category"""
        stats = {"collection": 0, "analysis": 0, "other": 0}
        for tool in self._tools.values():
            category = tool.category if tool.category in stats else "other"
            stats[category] += 1
        return stats

    # ========================================================================
    # Testing/Debugging Methods (Only used in tests, not in production)
    # ========================================================================
    
    def list_all_tools(self) -> List[str]:
        """
        List all registered tool names
        
        ⚠️ TEST ONLY: Used only in test_mcp_system.py for debugging
        """
        return sorted(self._tools.keys())
    
    def get_tools_by_server(self, server_name: str) -> List[ToolDefinition]:
        """
        Get all tools from a specific MCP server
        
        ⚠️ TEST ONLY: Used only in test_mcp_system.py for inspection
        """
        return [
            tool for tool in self._tools.values()
            if tool.server == server_name
        ]
    
    def get_mcp_tools(self) -> List[ToolDefinition]:
        """
        Get all tools from MCP servers (exclude legacy)
        
        ⚠️ TEST ONLY: Used only in get_tool_stats() for debugging
        """
        return [
            tool for tool in self._tools.values()
            if tool.server is not None
        ]
    
    def get_legacy_tools(self) -> List[ToolDefinition]:
        """
        Get all legacy tools (non-MCP)
        
        ⚠️ TEST ONLY: Used only in get_tool_stats() for debugging
        """
        return [
            tool for tool in self._tools.values()
            if tool.server is None
        ]
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about registered tools
        
        ⚠️ TEST ONLY: Used only in test_mcp_system.py for debugging
        """
        mcp_tools = self.get_mcp_tools()
        legacy_tools = self.get_legacy_tools()
        
        servers = {}
        for tool in mcp_tools:
            if tool.server not in servers:
                servers[tool.server] = 0
            servers[tool.server] += 1
        
        return {
            "total": len(self._tools),
            "mcp": len(mcp_tools),
            "legacy": len(legacy_tools),
            "servers": servers,
        }
