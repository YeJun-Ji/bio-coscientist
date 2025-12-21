"""
MCP (Model Context Protocol) Client
Core MCP protocol implementation for communicating with MCP servers via stdio
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Low-level MCP client that communicates with MCP servers via stdio.
    Implements JSON-RPC 2.0 protocol over stdin/stdout.
    
    Supports Node.js, Python, and other MCP server implementations.
    """
    
    def __init__(self, server_command: List[str], server_args: Optional[List[str]] = None, cwd: Optional[str] = None):
        """
        Initialize MCP client

        Args:
            server_command: Command to start the MCP server (e.g., ["node", "build/index.js"])
            server_args: Additional arguments for the server
            cwd: Working directory for the server process
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self.cwd = cwd
        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initialized = False
        self.server_name = " ".join(server_command)
        # Lock for thread-safe request/response handling
        # Prevents "readuntil() called while another coroutine is already waiting" error
        self._request_lock = asyncio.Lock()
        
    async def start(self):
        """Start the MCP server process and initialize connection"""
        try:
            full_command = self.server_command + self.server_args
            # Increase buffer limit to 10MB to handle large MCP responses (e.g., UniProt search results)
            # Default is 64KB which causes "Separator is not found, and chunk exceed the limit" errors
            self.process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,  # Working directory for servers that need it (e.g., pandas)
                limit=10 * 1024 * 1024  # 10MB buffer limit
            )
            self.logger.info(f"MCP server started: {' '.join(full_command)}")
            
            # Wait a moment for server to initialize
            await asyncio.sleep(0.5)
            
            # Send initialize request
            await self._initialize()
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def _initialize(self):
        """Initialize the MCP connection with capability negotiation"""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "biocoscientist",
                    "version": "1.0.0"
                }
            }
        }
        
        try:
            response = await self._send_request(init_request)
            self.initialized = True
            self.logger.info(f"MCP initialized successfully: {self.server_name}")
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            await self._send_notification(initialized_notification)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP: {e}")
            raise
    
    def _next_id(self) -> int:
        """Get next request ID for JSON-RPC"""
        self.request_id += 1
        return self.request_id
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send a notification (no response expected)"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not started")
        
        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        await self.process.stdin.drain()
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response (thread-safe with lock)"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not started")

        # Use lock to prevent concurrent access to stdin/stdout
        # This ensures only one request is processed at a time per MCP client
        async with self._request_lock:
            # Send request
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()

            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=30.0
                )
                if not response_line:
                    raise RuntimeError("MCP server closed connection")

                # Decode and strip whitespace
                response_str = response_line.decode().strip()
                if not response_str:
                    raise RuntimeError("MCP server returned empty response")

                response = json.loads(response_str)

                if "error" in response:
                    error_msg = response['error'].get('message', str(response['error']))
                    raise RuntimeError(f"MCP error: {error_msg}")

                return response.get("result", {})

            except asyncio.TimeoutError:
                raise RuntimeError("MCP request timeout")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server
        
        Returns:
            List of tool definitions with name, description, and inputSchema
        """
        if not self.initialized:
            raise RuntimeError("MCP client not initialized")
            
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        
        result = await self._send_request(request)
        tools = result.get("tools", [])
        self.logger.info(f"Listed {len(tools)} tools from {self.server_name}")
        return tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool (must match tool's inputSchema)
            
        Returns:
            Tool result content
        """
        if not self.initialized:
            raise RuntimeError("MCP client not initialized")
            
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        result = await self._send_request(request)
        return result.get("content", [])
    
    async def close(self):
        """Close the MCP server process gracefully"""
        if self.process:
            try:
                self.process.stdin.close()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.logger.info(f"MCP server closed: {self.server_name}")
