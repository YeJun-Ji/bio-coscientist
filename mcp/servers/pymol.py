"""
PyMol MCP Server Configuration
Provides access to PyMol molecular visualization and analysis
"""

import os
import subprocess
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PyMolServer:
    """
    PyMol MCP Server - Molecular visualization and analysis
    GitHub: https://github.com/vrtejus/pymol-mcp
    
    Provides tools for:
    - Protein structure visualization
    - Structure analysis (RMSD, distances)
    - Structure alignment
    - Surface and pocket analysis
    """
    
    name = "pymol"
    description = "PyMol molecular visualization and analysis"
    github_url = "https://github.com/vrtejus/pymol-mcp"
    
    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for PyMol server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/pymol-mcp")
    
    @staticmethod
    def is_installed() -> bool:
        """Check if PyMol server is installed"""
        install_path = PyMolServer.get_install_path()
        return os.path.exists(os.path.join(install_path, "pymol_mcp_server.py"))
    
    @staticmethod
    def install():
        """Install PyMol MCP server from GitHub"""
        install_path = PyMolServer.get_install_path()
        
        if PyMolServer.is_installed():
            logger.info(f"PyMol server already installed at {install_path}")
            return
        
        logger.info("Installing PyMol MCP server...")
        
        # Create directory
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        
        # Clone repository
        logger.info(f"Cloning from {PyMolServer.github_url}...")
        subprocess.run(
            ["git", "clone", PyMolServer.github_url, install_path],
            check=True,
            capture_output=True
        )
        
        # Install Python dependencies
        requirements_path = os.path.join(install_path, "requirements.txt")
        if os.path.exists(requirements_path):
            logger.info("Installing Python dependencies...")
            subprocess.run(
                ["pip", "install", "-r", requirements_path],
                cwd=install_path,
                check=True,
                capture_output=True
            )
        
        # Note: PyMol itself needs to be installed separately
        logger.warning(
            "⚠️  PyMol must be installed separately. "
            "Install via: conda install -c conda-forge pymol-open-source"
        )
        
        logger.info("✅ PyMol server installed successfully")
    
    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start PyMol server"""
        install_path = PyMolServer.get_install_path()
        server_script = "pymol_mcp_server.py"
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, server_script)

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"PyMol server not found at {server_path}. "
                "Please run PyMolServer.install() first."
            )

        # Use venv Python directly for better reliability
        if os.path.exists(venv_python):
            return [venv_python, server_path]

        # Fallback to uv run if venv not set up
        uv_path = os.path.expanduser("~/.local/bin/uv")
        return [uv_path, "run", "--python", "3.10", "--directory", install_path, "python", server_script]
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "name": PyMolServer.name,
            "description": PyMolServer.description,
            "command": PyMolServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
