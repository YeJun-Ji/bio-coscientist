"""
Reactome MCP Server Configuration
Provides access to Reactome pathway database
"""

import os
import subprocess
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ReactomeServer:
    """
    Reactome MCP Server - Biological pathway database
    GitHub: https://github.com/Augmented-Nature/Reactome-MCP-Server
    
    Provides tools for:
    - Pathway analysis
    - Protein-protein interactions
    - Biological processes
    - Disease pathways
    """
    
    name = "reactome"
    description = "Reactome biological pathway database"
    github_url = "https://github.com/Augmented-Nature/Reactome-MCP-Server"
    
    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Reactome server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/Reactome-MCP-Server")
    
    @staticmethod
    def is_installed() -> bool:
        """Check if Reactome server is installed"""
        install_path = ReactomeServer.get_install_path()
        index_path = os.path.join(install_path, "build", "index.js")
        return os.path.exists(index_path)
    
    @staticmethod
    def install():
        """Install Reactome MCP server from GitHub"""
        install_path = ReactomeServer.get_install_path()
        
        if ReactomeServer.is_installed():
            logger.info(f"Reactome server already installed at {install_path}")
            return
        
        logger.info("Installing Reactome MCP server...")
        
        # Create directory
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        
        # Clone repository
        logger.info(f"Cloning from {ReactomeServer.github_url}...")
        subprocess.run(
            ["git", "clone", ReactomeServer.github_url, install_path],
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
        
        logger.info("✅ Reactome server installed successfully")
    
    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Reactome server"""
        install_path = ReactomeServer.get_install_path()
        index_path = os.path.join(install_path, "build", "index.js")
        
        if not os.path.exists(index_path):
            raise RuntimeError(
                f"Reactome server not found at {index_path}. "
                "Please run ReactomeServer.install() first."
            )
        
        return ["node", index_path]
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "name": ReactomeServer.name,
            "description": ReactomeServer.description,
            "command": ReactomeServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",
        }
