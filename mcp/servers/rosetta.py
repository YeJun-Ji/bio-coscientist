"""
Rosetta MCP Server Configuration
Provides access to Rosetta protein design and modeling tools
"""

import os
import subprocess
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RosettaServer:
    """
    Rosetta MCP Server - Protein design and modeling
    GitHub: https://github.com/Arielbs/rosetta-mcp-server
    
    Provides tools for:
    - Protein structure prediction
    - Protein design and optimization
    - Docking simulations
    - Energy calculations
    """
    
    name = "rosetta"
    description = "Rosetta protein design and modeling toolkit"
    github_url = "https://github.com/Arielbs/rosetta-mcp-server"
    
    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Rosetta server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/rosetta-mcp-server")
    
    @staticmethod
    def is_installed() -> bool:
        """Check if Rosetta server is installed"""
        install_path = RosettaServer.get_install_path()
        # Check for Node.js MCP wrapper
        return os.path.exists(os.path.join(install_path, "rosetta_mcp_wrapper.js"))
    
    @staticmethod
    def install():
        """Install Rosetta MCP server from GitHub"""
        install_path = RosettaServer.get_install_path()
        
        if RosettaServer.is_installed():
            logger.info(f"Rosetta server already installed at {install_path}")
            return
        
        logger.info("Installing Rosetta MCP server...")
        
        # Create directory
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        
        # Clone repository
        logger.info(f"Cloning from {RosettaServer.github_url}...")
        subprocess.run(
            ["git", "clone", RosettaServer.github_url, install_path],
            check=True,
            capture_output=True
        )
        
        # Install Python dependencies if requirements.txt exists
        requirements_path = os.path.join(install_path, "requirements.txt")
        if os.path.exists(requirements_path):
            logger.info("Installing Python dependencies...")
            subprocess.run(
                ["pip", "install", "-r", requirements_path],
                cwd=install_path,
                check=True,
                capture_output=True
            )
        
        logger.info("✅ Rosetta server installed successfully")
    
    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Rosetta server"""
        install_path = RosettaServer.get_install_path()
        # Use the Node.js MCP wrapper that wraps the Python server
        wrapper_path = os.path.join(install_path, "rosetta_mcp_wrapper.js")

        if not os.path.exists(wrapper_path):
            raise RuntimeError(
                f"Rosetta server not found at {wrapper_path}. "
                "Please run RosettaServer.install() first."
            )

        return ["node", wrapper_path]
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "name": RosettaServer.name,
            "description": RosettaServer.description,
            "command": RosettaServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
