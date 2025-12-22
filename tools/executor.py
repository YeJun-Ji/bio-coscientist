"""
Tool Executor - Executes tool calls at runtime (Dynamic MCP Routing)

This module routes tool calls to appropriate handlers:
1. MCP Server tools -> routed to MCPServerManager
2. Legacy tools -> routed to client methods (deprecated)

Supports both sequential and parallel execution modes.

NEW (v3.2): Automatic parameter mapping via ParameterMapper
- No per-tool configuration required
- Pattern-based parameter resolution from collected data
- Schema-driven validation and type coercion
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
import json

from .parameter_mapper import ParameterMapper, create_mapper_context

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tool calls by routing to MCP servers or legacy clients

    Architecture:
    1. LLM selects a tool via function calling
    2. ToolExecutor receives the tool name and arguments
    3. ParameterMapper auto-fills missing parameters from context (NEW)
    4. ToolRegistry provides tool metadata (which server, etc.)
    5. For MCP tools: route to MCPServerManager
    6. For legacy tools: route to old client methods (deprecated)
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

        # NEW: Parameter mapper for automatic argument resolution
        self.parameter_mapper = ParameterMapper()

        # NEW: Context for parameter mapping (set via set_context)
        self._context: Dict[str, Any] = {
            "input_files": {},
            "collected_data": {},
            "entity_analysis": {}
        }

    def set_context(
        self,
        input_files: Dict[str, Any] = None,
        collected_data: Dict[str, Any] = None,
        entity_analysis: Dict[str, Any] = None
    ):
        """
        Set context for automatic parameter mapping.

        Call this before executing tools to enable automatic
        parameter resolution from collected data.

        Args:
            input_files: Input data files {name: {path, type, ...}}
            collected_data: Data from Chain 1 collection
            entity_analysis: Entity analysis from Stage 1
        """
        if input_files is not None:
            self._context["input_files"] = input_files
        if collected_data is not None:
            self._context["collected_data"] = collected_data
        if entity_analysis is not None:
            self._context["entity_analysis"] = entity_analysis

        self.logger.debug(
            f"Context updated: {len(self._context['input_files'])} files, "
            f"{len(self._context['collected_data'])} sources, "
            f"{len(self._context.get('entity_analysis', {}).get('primary_entities', []))} entities"
        )
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a single tool call with automatic parameter mapping.

        NEW FEATURES (v3.2):
        1. Automatic parameter mapping from context (collected_data, input_files)
        2. File path arguments (*_file) auto-converted to content
        3. Schema-based validation with helpful error messages

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
        self.logger.debug(f"  Original arguments: {list(arguments.keys())}")

        try:
            # === STEP 1: Automatic parameter mapping (NEW v3.2) ===
            if self._context and any(self._context.values()):
                arguments, warnings = self.parameter_mapper.map_parameters(
                    tool_name=tool_name,
                    tool_schema=tool_def.parameters,
                    provided_args=arguments,
                    context=self._context
                )
                if warnings:
                    for warning in warnings:
                        self.logger.warning(f"  ⚠️ {warning}")
                self.logger.debug(f"  After mapping: {list(arguments.keys())}")

            # === STEP 2: File argument preprocessing (legacy) ===
            arguments = await self._preprocess_file_arguments(tool_name, arguments)

            # === STEP 3: Validate required parameters ===
            self._validate_required_parameters(tool_def, arguments)

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
        arguments: Dict[str, Any],
        max_retries: int = 3
    ) -> Any:
        """
        Execute a tool via MCP server with automatic retry on transient failures.

        Retry is triggered when:
        - Server returns empty response (recovered via restart)
        - JSON parsing fails (recovered via restart)
        - Server connection closes unexpectedly
        """
        if not self.mcp_manager:
            raise RuntimeError(
                f"MCP server required for tool '{tool_def.name}' "
                "but MCPServerManager not available"
            )

        self.logger.debug(f"Routing to MCP server: {tool_def.server}")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = await self.mcp_manager.call_tool(tool_def.name, arguments)
                return result
            except RuntimeError as e:
                error_msg = str(e).lower()
                # Check if this is a recoverable error (server restarted)
                if "retry" in error_msg or "restarted" in error_msg or "empty response" in error_msg:
                    last_error = e
                    if attempt < max_retries:
                        self.logger.warning(
                            f"  ⚠️ MCP call failed (attempt {attempt}/{max_retries}): {e}"
                        )
                        # Wait briefly before retry to let server stabilize
                        await asyncio.sleep(0.5 * attempt)
                        continue
                # Non-recoverable error, raise immediately
                raise

        # All retries exhausted
        raise RuntimeError(f"MCP tool '{tool_def.name}' failed after {max_retries} retries: {last_error}")
    
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

    async def _preprocess_file_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preprocess file path arguments by loading and converting to strings.

        Handles patterns like:
        - sequence_file → sequence (string)
        - structure_file → structure (PDB string)
        - data_file → data (JSON parsed)

        Examples:
            esmfold_predict(sequence_file="data/req_1/collection/sources.json")
            → Load file, extract sequence field
            → esmfold_predict(sequence="MKTAYIAKQR...")

        Args:
            tool_name: Name of the tool being executed
            arguments: Original arguments dict

        Returns:
            Processed arguments dict with file params converted to strings
        """
        processed = arguments.copy()

        # File parameter patterns: (file_param, target_param)
        file_params = [
            ("sequence_file", "sequence"),
            ("structure_file", "structure"),
            ("data_file", "data"),
        ]

        for file_param, string_param in file_params:
            if file_param in processed:
                file_path = processed[file_param]

                try:
                    # Load file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)

                    # Convert based on tool requirements
                    if file_param == "sequence_file":
                        # Extract sequence from collected data structure
                        processed[string_param] = self._extract_sequence(file_data, tool_name)
                    elif file_param == "structure_file":
                        processed[string_param] = file_data  # Already string for PDB
                    elif file_param == "data_file":
                        processed[string_param] = file_data

                    # Remove file parameter
                    del processed[file_param]

                    self.logger.info(f"[ToolExecutor] Loaded {file_param}: {file_path} → {string_param}")

                except Exception as e:
                    self.logger.error(f"[ToolExecutor] Failed to load {file_param}: {e}")
                    # Keep original argument, let tool handle error

        return processed

    def _extract_sequence(self, file_data: Dict, tool_name: str) -> str:
        """
        Extract sequence string from collected data file.

        Handles various source formats:
        - UniProt: file_data["UniProt"][0]["result"]["sequence"]
        - Direct: file_data["sequence"]
        - FASTA: file_data["fasta"]

        Args:
            file_data: Loaded JSON data from sources.json
            tool_name: Name of the tool requesting sequence

        Returns:
            Extracted sequence string, or "" if not found
        """
        # Strategy 1: Direct sequence field
        if "sequence" in file_data:
            return file_data["sequence"]

        # Strategy 2: UniProt source
        if "UniProt" in file_data and isinstance(file_data["UniProt"], list):
            for item in file_data["UniProt"]:
                result = item.get("result", {})
                if "sequence" in result:
                    self.logger.debug(f"Extracted sequence from UniProt source (length: {len(result['sequence'])})")
                    return result["sequence"]

        # Strategy 3: First source with sequence
        for source_name, source_data in file_data.items():
            if isinstance(source_data, list):
                for item in source_data:
                    result = item.get("result", {})
                    if "sequence" in result:
                        self.logger.debug(f"Extracted sequence from {source_name} (length: {len(result['sequence'])})")
                        return result["sequence"]

        # Fallback: Return empty string (tool will handle validation)
        self.logger.warning(f"[ToolExecutor] Could not extract sequence from file for {tool_name}")
        return ""

    def _validate_required_parameters(
        self,
        tool_def,
        arguments: Dict[str, Any]
    ) -> None:
        """
        Validate that all required parameters are provided.

        Args:
            tool_def: ToolDefinition object
            arguments: Arguments dict to validate

        Raises:
            ValueError: If required parameters are missing
        """
        parameters_schema = tool_def.parameters
        required_params = parameters_schema.get("required", [])

        if not required_params:
            return  # No required parameters

        missing_params = [param for param in required_params if param not in arguments]

        if missing_params:
            # Extract parameter details from schema for helpful error message
            properties = parameters_schema.get("properties", {})
            missing_details = []
            for param in missing_params:
                param_schema = properties.get(param, {})
                param_type = param_schema.get("type", "unknown")
                param_desc = param_schema.get("description", "")
                missing_details.append(
                    f"  - {param} ({param_type}): {param_desc}"
                )

            error_msg = (
                f"Tool '{tool_def.name}' missing required parameters:\n"
                + "\n".join(missing_details)
                + f"\n\nProvided parameters: {list(arguments.keys())}"
                + f"\nRequired parameters: {required_params}"
            )

            self.logger.error(f"  ✗ Parameter validation failed:\n{error_msg}")
            raise ValueError(error_msg)
