"""
Tool Executor - Executes tool calls at runtime (Dynamic MCP Routing)

This module routes tool calls to appropriate handlers:
1. MCP Server tools -> routed to MCPServerManager
2. Legacy tools -> routed to client methods (deprecated)

Supports both sequential and parallel execution modes.
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tool calls by routing to MCP servers or legacy clients
    
    Architecture:
    1. LLM selects a tool via function calling
    2. ToolExecutor receives the tool name and arguments
    3. ToolRegistry provides tool metadata (which server, etc.)
    4. For MCP tools: route to MCPServerManager
    5. For legacy tools: route to old client methods (deprecated)
    """
    
    def __init__(
        self,
        tool_registry,
        mcp_server_manager=None,
        legacy_clients: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tool executor
        
        Args:
            tool_registry: ToolRegistry instance
            mcp_server_manager: MCPServerManager for MCP tool routing
            legacy_clients: Dict of legacy client instances (deprecated)
        """
        self.registry = tool_registry
        self.mcp_manager = mcp_server_manager
        self.legacy_clients = legacy_clients or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a single tool call
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
        
        Returns:
            Result from tool execution
        """
        tool_def = self.registry.get_tool(tool_name)
        
        if not tool_def:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        self.logger.info(f"Executing tool: {tool_name}")
        
        try:
            # Route to MCP server if available
            if tool_def.server:
                result = await self._execute_mcp_tool(tool_def, arguments)
            else:
                # Legacy tool routing (deprecated)
                result = await self._execute_legacy_tool(tool_def, arguments)
            
            self.logger.info(f"  ✓ Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"  ✗ Tool {tool_name} execution failed: {e}")
            raise
    
    async def _execute_mcp_tool(
        self,
        tool_def,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool via MCP server"""
        if not self.mcp_manager:
            raise RuntimeError(
                f"MCP server required for tool '{tool_def.name}' "
                "but MCPServerManager not available"
            )
        
        self.logger.debug(f"Routing to MCP server: {tool_def.server}")
        return await self.mcp_manager.call_tool(tool_def.name, arguments)
    
    async def _execute_legacy_tool(
        self,
        tool_def,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a legacy tool via client method (deprecated)"""
        client_type = tool_def.metadata.get("client_type")
        method_name = tool_def.metadata.get("method_name")
        
        if not client_type or not method_name:
            raise ValueError(
                f"Legacy tool '{tool_def.name}' missing client_type or method_name"
            )
        
        client = self.legacy_clients.get(client_type)
        if not client:
            raise ValueError(f"Client not available: {client_type}")
        
        method = getattr(client, method_name, None)
        if not method:
            raise ValueError(
                f"Method {method_name} not found on {client_type}"
            )
        
        self.logger.warning(
            f"Using deprecated legacy tool: {tool_def.name}. "
            "Consider migrating to MCP server."
        )
        
        return await method(**arguments)
    
    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls and return formatted results.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'
            parallel: If True, execute tools in parallel using asyncio.gather.
                     If False, execute sequentially (original behavior).
                     Default is True for optimal performance.

        Returns:
            List of results with 'name', 'arguments', 'result', and 'status'
        """
        if not tool_calls:
            return []

        if parallel:
            return await self._execute_parallel(tool_calls)
        else:
            return await self._execute_sequential(tool_calls)

    async def _execute_sequential(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls sequentially (original behavior)"""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            try:
                result = await self.execute_tool(tool_name, arguments)

                results.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                self.logger.error(f"Tool execution failed: {tool_name}: {e}")
                results.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": None,
                    "error": str(e),
                    "status": "error"
                })

        return results

    async def _execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls in parallel using asyncio.gather.

        This provides significant performance improvements when multiple
        independent tools need to be called (e.g., fetching data from
        different MCP servers simultaneously).
        """
        self.logger.info(f"Executing {len(tool_calls)} tools in parallel")

        # Create coroutines for all tool calls
        async def execute_single(tool_call: Dict[str, Any]) -> Dict[str, Any]:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            try:
                result = await self.execute_tool(tool_name, arguments)
                return {
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "status": "success"
                }
            except Exception as e:
                self.logger.error(f"Tool execution failed: {tool_name}: {e}")
                return {
                    "name": tool_name,
                    "arguments": arguments,
                    "result": None,
                    "error": str(e),
                    "status": "error"
                }

        # Execute all in parallel
        results = await asyncio.gather(
            *[execute_single(tc) for tc in tool_calls],
            return_exceptions=True
        )

        # Handle any unexpected exceptions from gather itself
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # This shouldn't happen since execute_single catches exceptions,
                # but handle it just in case
                tool_name = tool_calls[i].get("name", "unknown")
                self.logger.error(f"Unexpected error in parallel execution: {tool_name}: {result}")
                processed_results.append({
                    "name": tool_name,
                    "arguments": tool_calls[i].get("arguments", {}),
                    "result": None,
                    "error": str(result),
                    "status": "error"
                })
            else:
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.get("status") == "success")
        self.logger.info(f"Parallel execution complete: {successful}/{len(tool_calls)} successful")

        return processed_results

    async def execute_tool_calls_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls in parallel batches.

        Useful when you have many tool calls but want to limit
        concurrent requests to avoid overwhelming servers.

        Args:
            tool_calls: List of tool call dicts
            batch_size: Maximum number of concurrent executions

        Returns:
            List of results from all batches
        """
        if not tool_calls:
            return []

        all_results = []

        for i in range(0, len(tool_calls), batch_size):
            batch = tool_calls[i:i + batch_size]
            self.logger.info(f"Executing batch {i // batch_size + 1}: {len(batch)} tools")

            batch_results = await self._execute_parallel(batch)
            all_results.extend(batch_results)

        return all_results
