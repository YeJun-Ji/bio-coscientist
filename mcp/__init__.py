"""
MCP (Model Context Protocol) Module
Provides unified access to multiple MCP servers for biological data and tools
"""

from .client import MCPClient
from .server_manager import MCPServerManager, ServerConfig

__all__ = [
    "MCPClient",
    "MCPServerManager",
    "ServerConfig",
]
