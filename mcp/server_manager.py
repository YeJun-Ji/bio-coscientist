"""
MCP Server Manager
Manages multiple MCP servers and provides dynamic tool loading
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .client import MCPClient
from .servers import (
    # Original servers (7)
    KEGGServer,
    RosettaServer,
    NCBIServer,
    ESMFoldServer,
    UniProtServer,
    BLASTServer,
    PandasAnalysisServer,
    # New servers - v1 (5)
    STRINGDBServer,
    ChEMBLServer,
    NanoporeServer,
    MSAServer,
    FoldseekServer,
    # New servers - v2 (10) - Problem 3 & 4 support
    RCSBPDBServer,
    InterProServer,
    GProfilerServer,
    OpenTargetsServer,
    NetworkXServer,
    IEDBServer,
    VinaServer,
    ProteinMPNNServer,
    RFdiffusionServer,
    ColabFoldServer,
    # Removed servers (3) - commented for reference
    # ReactomeServer,  # REMOVED: KEGG covers pathway + drug/compound functionality
    # PyMolServer,  # REMOVED: Local visualization tool, not needed for automation
    # ScholarServer,  # REMOVED: NCBI PubMed covers literature search
)

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for an MCP server"""
    name: str
    description: str
    command: List[str]
    args: List[str]
    auto_install: bool
    problem_types: List[str]
    stages: List[str]
    category: str = "collection"  # "collection" | "analysis"
    enabled: bool = True
    cwd: Optional[str] = None  # Working directory for server process


class MCPServerManager:
    """
    Manages multiple MCP servers and provides unified tool access
    
    Architecture:
    1. Each MCP Server (KEGG, Rosetta, etc.) is an external process
    2. Each server provides multiple tools via MCP protocol
    3. ServerManager discovers all tools from all servers dynamically
    4. Tools are registered in ToolRegistry for LLM function calling
    5. ToolExecutor routes tool calls back to appropriate server
    """
    
    # Available server classes
    # Note: Reactome, PyMol, Scholar removed in MCP consolidation
    # Reactome → KEGG covers pathway + drug/compound
    # PyMol → Local visualization, not needed for automation
    # Scholar → NCBI PubMed covers literature search
    # Server configuration: 22 total (7 original + 5 v1 + 10 v2)
    # Problem coverage:
    # - P1(MSA, Foldseek), P2(Nanopore), P3(Rosetta+BLAST+RFdiffusion+ProteinMPNN+ColabFold)
    # - P4(STRING+NetworkX+IEDB+OpenTargets+RCSB), P5(ChEMBL+STRING)
    SERVER_CLASSES = {
        # Data Collection (4)
        "kegg": KEGGServer,           # Pathways, drugs, compounds, diseases
        "ncbi": NCBIServer,           # PubMed, GenBank, gene info
        "uniprot": UniProtServer,     # Protein sequences, annotations
        "rcsbpdb": RCSBPDBServer,     # PDB structure download, binding sites (P3, P4)

        # Data Analysis - Structure (4)
        "esmfold": ESMFoldServer,     # Protein structure prediction
        "rosetta": RosettaServer,     # Docking, energy calculation (P3)
        "blast": BLASTServer,         # Sequence similarity, off-target (P3)
        "foldseek": FoldseekServer,   # Structure similarity search (P1 Req.5)

        # Data Analysis - Networks & Drugs (4)
        "stringdb": STRINGDBServer,   # PPI networks (P4, P5)
        "chembl": ChEMBLServer,       # Drug-target data (P5)
        "networkx": NetworkXServer,   # Network analysis, hubs, communities (P4)
        "opentargets": OpenTargetsServer,  # Druggability, target-drug (P4, P5)

        # Data Analysis - Specialized (6)
        "pandas_analysis": PandasAnalysisServer,  # CSV/data analysis
        "nanopore": NanoporeServer,   # poly(A) analysis (P2)
        "msa": MSAServer,             # Multiple sequence alignment (P1)
        "interpro": InterProServer,   # Domain analysis (P3, P4)
        "gprofiler": GProfilerServer, # GO/Pathway enrichment (P4)
        "iedb": IEDBServer,           # MHC binding, immunogenicity (P3, P4)

        # De novo Design - GPU required (4)
        "vina": VinaServer,           # Molecular docking (P3, P5)
        "proteinmpnn": ProteinMPNNServer,  # Sequence design for backbone (P3) - GPU 8GB+
        "rfdiffusion": RFdiffusionServer,  # Binder backbone design (P3) - GPU 16GB+
        "colabfold": ColabFoldServer,      # Complex structure prediction (P3, P4) - GPU 24GB+

        # Removed servers (kept for reference)
        # "reactome": ReactomeServer,  # → KEGG
        # "pymol": PyMolServer,        # → Local tool
        # "scholar": ScholarServer,    # → NCBI PubMed
    }
    
    def __init__(self, enabled_servers: Optional[List[str]] = None):
        """
        Initialize MCP Server Manager
        
        Args:
            enabled_servers: List of server names to enable (default: all)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Determine which servers to enable
        if enabled_servers is None:
            # Enable all by default
            enabled_servers = list(self.SERVER_CLASSES.keys())
        
        self.enabled_servers = enabled_servers
        self.clients: Dict[str, MCPClient] = {}
        self.server_configs: Dict[str, ServerConfig] = {}
        self.tools_by_server: Dict[str, List[Dict[str, Any]]] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize all enabled MCP servers"""
        self.logger.info(f"Initializing {len(self.enabled_servers)} MCP servers...")
        
        for server_name in self.enabled_servers:
            try:
                await self._initialize_server(server_name)
            except Exception as e:
                self.logger.error(f"Failed to initialize {server_name} server: {e}")
                # Continue with other servers
        
        self.initialized = True
        total_tools = sum(len(tools) for tools in self.tools_by_server.values())
        self.logger.info(
            f"✅ MCP initialization complete: "
            f"{len(self.clients)} servers, {total_tools} tools"
        )
    
    async def _initialize_server(self, server_name: str):
        """Initialize a single MCP server"""
        server_class = self.SERVER_CLASSES.get(server_name)
        if not server_class:
            raise ValueError(f"Unknown server: {server_name}")
        
        self.logger.info(f"Setting up {server_name} server...")
        
        # Get server configuration
        config_dict = server_class.get_config()
        config = ServerConfig(
            name=config_dict["name"],
            description=config_dict["description"],
            command=config_dict["command"],
            args=config_dict.get("args", []),
            auto_install=config_dict.get("auto_install", True),
            problem_types=config_dict.get("problem_types", ["all"]),
            stages=config_dict.get("stages", ["all"]),
            category=config_dict.get("category", "collection"),
            cwd=config_dict.get("cwd"),  # Working directory for servers with local imports
        )
        
        # Install if needed
        if config.auto_install and not server_class.is_installed():
            self.logger.info(f"Installing {server_name} server...")
            try:
                server_class.install()
            except Exception as e:
                self.logger.error(f"Failed to install {server_name}: {e}")
                raise
        
        # Create and start MCP client
        client = MCPClient(config.command, config.args, cwd=config.cwd)
        await client.start()
        
        # List available tools
        tools = await client.list_tools()
        
        self.clients[server_name] = client
        self.server_configs[server_name] = config
        self.tools_by_server[server_name] = tools
        
        self.logger.info(
            f"✅ {server_name} server ready: {len(tools)} tools available"
        )
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools from all servers in OpenAI function format
        
        Returns:
            List of tool definitions for LLM function calling
        """
        all_tools = []
        
        for server_name, tools in self.tools_by_server.items():
            for tool in tools:
                # Convert MCP tool to OpenAI function format
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {}),
                    },
                    # Add metadata for routing
                    "_server": server_name,
                    "_config": self.server_configs[server_name],
                }
                all_tools.append(tool_def)
        
        return all_tools
    
    def get_tools_for_problem(
        self,
        problem_type: str,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools filtered by problem type and stage
        
        Args:
            problem_type: Problem type (e.g., "protein_binder_design")
            stage: Optional stage filter (e.g., "generation")
        
        Returns:
            Filtered list of tool definitions
        """
        all_tools = self.get_all_tools()
        filtered = []
        
        for tool in all_tools:
            config = tool["_config"]
            
            # Check problem type
            if "all" not in config.problem_types and \
               problem_type not in config.problem_types:
                continue
            
            # Check stage
            if stage and "all" not in config.stages and \
               stage not in config.stages:
                continue
            
            filtered.append(tool)
        
        return filtered
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on its appropriate server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
        
        Returns:
            Tool result
        """
        # Find which server has this tool
        server_name = None
        for srv_name, tools in self.tools_by_server.items():
            if any(t["name"] == tool_name for t in tools):
                server_name = srv_name
                break
        
        if not server_name:
            raise ValueError(f"Tool not found: {tool_name}")
        
        client = self.clients.get(server_name)
        if not client:
            raise RuntimeError(f"Server not initialized: {server_name}")
        
        self.logger.info(f"Calling {tool_name} on {server_name} server")
        return await client.call_tool(tool_name, arguments)
    
    async def close(self):
        """Close all MCP server connections"""
        self.logger.info("Closing MCP servers...")
        
        for server_name, client in self.clients.items():
            try:
                await client.close()
                self.logger.info(f"Closed {server_name} server")
            except Exception as e:
                self.logger.warning(f"Error closing {server_name} server: {e}")
        
        self.clients.clear()
        self.tools_by_server.clear()
        self.initialized = False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about all connected servers"""
        info = {}
        
        for server_name, config in self.server_configs.items():
            tool_count = len(self.tools_by_server.get(server_name, []))
            info[server_name] = {
                "description": config.description,
                "tool_count": tool_count,
                "problem_types": config.problem_types,
                "stages": config.stages,
                "category": config.category,
                "status": "connected" if server_name in self.clients else "disconnected",
            }
        
        return info
