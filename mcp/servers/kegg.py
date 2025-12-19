"""
KEGG MCP Server Configuration
Provides access to KEGG pathway, gene, and compound databases
"""

import os
import subprocess
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KEGGServer:
    """
    KEGG MCP Server - Pathway and molecular interaction database
    GitHub: https://github.com/Augmented-Nature/KEGG-MCP-Server
    
    Provides tools for:
    - Pathway search and analysis
    - Gene and protein information
    - Compound and drug data
    - Disease associations
    """
    
    name = "kegg"
    description = "KEGG pathway and molecular interaction database"
    github_url = "https://github.com/Augmented-Nature/KEGG-MCP-Server"
    
    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for KEGG server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/KEGG-MCP-Server")
    
    @staticmethod
    def is_installed() -> bool:
        """Check if KEGG server is installed"""
        install_path = KEGGServer.get_install_path()
        index_path = os.path.join(install_path, "build", "index.js")
        return os.path.exists(index_path)
    
    @staticmethod
    def install():
        """Install KEGG MCP server from GitHub"""
        install_path = KEGGServer.get_install_path()
        
        if KEGGServer.is_installed():
            logger.info(f"KEGG server already installed at {install_path}")
            return
        
        logger.info("Installing KEGG MCP server...")
        
        # Create directory
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        
        # Clone repository
        logger.info(f"Cloning from {KEGGServer.github_url}...")
        subprocess.run(
            ["git", "clone", KEGGServer.github_url, install_path],
            check=True,
            capture_output=True
        )
        
        # Install dependencies
        logger.info("Installing npm dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=install_path,
            check=True,
            capture_output=True
        )
        
        # Build server
        logger.info("Building server...")
        subprocess.run(
            ["npm", "run", "build"],
            cwd=install_path,
            check=True,
            capture_output=True
        )
        
        logger.info("✅ KEGG server installed successfully")
    
    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start KEGG server"""
        install_path = KEGGServer.get_install_path()
        index_path = os.path.join(install_path, "build", "index.js")
        
        if not os.path.exists(index_path):
            raise RuntimeError(
                f"KEGG server not found at {index_path}. "
                "Please run KEGGServer.install() first."
            )
        
        return ["node", index_path]
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "name": KEGGServer.name,
            "description": KEGGServer.description,
            "command": KEGGServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",
        }
