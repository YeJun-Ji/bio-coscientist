"""
NCBI MCP Server Configuration
Provides access to NCBI databases (PubMed, GenBank, etc.)
"""

import os
import subprocess
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class NCBIServer:
    """
    NCBI MCP Server - National Center for Biotechnology Information databases
    GitHub: https://github.com/vitorpavinato/ncbi-mcp-server
    
    Provides tools for:
    - PubMed literature search
    - GenBank sequence retrieval
    - Protein database queries
    - Gene information
    """
    
    name = "ncbi"
    description = "NCBI databases (PubMed, GenBank, Protein, Gene)"
    github_url = "https://github.com/vitorpavinato/ncbi-mcp-server"
    
    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for NCBI server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/ncbi-mcp-server")
    
    @staticmethod
    def is_installed() -> bool:
        """Check if NCBI server is installed"""
        install_path = NCBIServer.get_install_path()
        return os.path.exists(os.path.join(install_path, "src", "ncbi_mcp_server", "server.py"))
    
    @staticmethod
    def install():
        """Install NCBI MCP server from GitHub"""
        install_path = NCBIServer.get_install_path()
        
        if NCBIServer.is_installed():
            logger.info(f"NCBI server already installed at {install_path}")
            return
        
        logger.info("Installing NCBI MCP server...")
        
        # Create directory
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        
        # Clone repository
        logger.info(f"Cloning from {NCBIServer.github_url}...")
        subprocess.run(
            ["git", "clone", NCBIServer.github_url, install_path],
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
        
        # Check for package installation
        if os.path.exists(os.path.join(install_path, "setup.py")):
            logger.info("Installing package...")
            subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=install_path,
                check=True,
                capture_output=True
            )
        
        logger.info("✅ NCBI server installed successfully")
    
    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start NCBI server"""
        install_path = NCBIServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "src", "ncbi_mcp_server", "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"NCBI server not found at {server_path}. "
                "Please run NCBIServer.install() first."
            )

        # Use venv Python directly (package installed via uv pip install -e .)
        if os.path.exists(venv_python):
            return [venv_python, "-m", "ncbi_mcp_server.server"]

        # Fallback to uv run (will auto-install dependencies)
        uv_path = os.path.expanduser("~/.local/bin/uv")
        return [
            uv_path, "run",
            "--python", "3.10",
            "--directory", install_path,
            "python", "-m", "ncbi_mcp_server.server"
        ]
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "name": NCBIServer.name,
            "description": NCBIServer.description,
            "command": NCBIServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",
        }
