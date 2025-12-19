"""
Google Scholar MCP Server Configuration
Provides access to Google Scholar for academic literature search
"""

import os
import subprocess
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ScholarServer:
    """
    Google Scholar MCP Server - Academic literature search
    GitHub: https://github.com/JackKuo666/Google-Scholar-MCP-Server
    
    Provides tools for:
    - Academic paper search
    - Citation analysis
    - Author profiles
    - Related paper discovery
    """
    
    name = "scholar"
    description = "Google Scholar academic literature search"
    github_url = "https://github.com/JackKuo666/Google-Scholar-MCP-Server"
    
    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Scholar server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/Google-Scholar-MCP-Server")
    
    @staticmethod
    def is_installed() -> bool:
        """Check if Scholar server is installed"""
        install_path = ScholarServer.get_install_path()
        return os.path.exists(os.path.join(install_path, "google_scholar_server.py"))
    
    @staticmethod
    def install():
        """Install Google Scholar MCP server from GitHub"""
        install_path = ScholarServer.get_install_path()
        
        if ScholarServer.is_installed():
            logger.info(f"Scholar server already installed at {install_path}")
            return
        
        logger.info("Installing Google Scholar MCP server...")
        
        # Create directory
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        
        # Clone repository
        logger.info(f"Cloning from {ScholarServer.github_url}...")
        subprocess.run(
            ["git", "clone", ScholarServer.github_url, install_path],
            check=True,
            capture_output=True
        )
        
        # Check if it's a Node.js project
        if os.path.exists(os.path.join(install_path, "package.json")):
            logger.info("Installing npm dependencies...")
            subprocess.run(
                ["npm", "install"],
                cwd=install_path,
                check=True,
                capture_output=True
            )
            
            # Build if there's a build script
            package_json_path = os.path.join(install_path, "package.json")
            import json
            with open(package_json_path) as f:
                package_data = json.load(f)
                if "build" in package_data.get("scripts", {}):
                    logger.info("Building server...")
                    subprocess.run(
                        ["npm", "run", "build"],
                        cwd=install_path,
                        check=True,
                        capture_output=True
                    )
        
        # Check if it's a Python project
        requirements_path = os.path.join(install_path, "requirements.txt")
        if os.path.exists(requirements_path):
            logger.info("Installing Python dependencies...")
            subprocess.run(
                ["pip", "install", "-r", requirements_path],
                cwd=install_path,
                check=True,
                capture_output=True
            )
        
        logger.info("✅ Scholar server installed successfully")
    
    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Scholar server"""
        install_path = ScholarServer.get_install_path()
        server_script = "google_scholar_server.py"
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, server_script)

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"Scholar server not found at {server_path}. "
                "Please run ScholarServer.install() first."
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
            "name": ScholarServer.name,
            "description": ScholarServer.description,
            "command": ScholarServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",
        }
