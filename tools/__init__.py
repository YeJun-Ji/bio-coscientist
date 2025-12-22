"""
Tools package - Dynamic tool management for MCP servers

This package provides:
- ToolRegistry: Dynamic tool discovery and registration from MCP servers
- ToolExecutor: Routes tool calls to appropriate MCP servers
- ToolDefinition: Tool metadata structure
- ParameterMapper: Automatic parameter resolution from context (NEW v3.2)

All tools are automatically discovered from MCP servers at runtime.
"""

from .registry import ToolRegistry, ToolDefinition
from .executor import ToolExecutor
from .parameter_mapper import ParameterMapper, create_mapper_context

__all__ = ['ToolRegistry', 'ToolDefinition', 'ToolExecutor', 'ParameterMapper', 'create_mapper_context']
