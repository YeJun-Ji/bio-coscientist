"""
NetworkX MCP Server Configuration
Provides PPI network analysis using NetworkX library

NetworkX provides:
- Network construction from PPI data
- Centrality analysis (degree, betweenness, closeness)
- Community detection
- Shortest path analysis
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class NetworkXServer:
    """
    NetworkX MCP Server - PPI Network Analysis

    Provides tools for:
    - build_network: Construct network from STRING/BioGRID data
    - find_hub_proteins: Identify hub proteins by centrality
    - find_communities: Detect network communities/clusters
    - calculate_centrality: Compute various centrality metrics
    - shortest_path: Find shortest path between proteins

    Used for:
    - Problem 4: IL-11 interactor network analysis
    - Hub protein identification for target prioritization
    """

    name = "networkx"
    description = "NetworkX PPI network analysis - centrality, communities, paths"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for NetworkX server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/networkx-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if NetworkX server is installed"""
        install_path = NetworkXServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install NetworkX MCP server from template"""
        install_path = NetworkXServer.get_install_path()

        if NetworkXServer.is_installed():
            logger.info(f"NetworkX server already installed at {install_path}")
            return

        logger.info(f"Installing NetworkX server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("networkx", install_path):
                raise RuntimeError("Failed to install NetworkX server template")
            logger.info("✅ Installed server.py from template")

        # Create venv and install dependencies
        venv_path = os.path.join(install_path, ".venv")
        venv_python = os.path.join(venv_path, "bin", "python")
        uv_path = os.path.expanduser("~/.local/bin/uv")

        if not os.path.exists(venv_path):
            logger.info("Creating virtual environment...")
            subprocess.run(
                [uv_path, "venv", "--python", "3.10", ".venv"],
                cwd=install_path,
                check=True,
                capture_output=True
            )

        # Install dependencies using uv pip
        logger.info("Installing dependencies...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "mcp", "requests", "networkx", "python-louvain"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ NetworkX server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start NetworkX server"""
        install_path = NetworkXServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"NetworkX server not found at {server_path}. "
                "Please run NetworkXServer.install() first."
            )

        if os.path.exists(venv_python):
            return [venv_python, server_path]

        uv_path = os.path.expanduser("~/.local/bin/uv")
        return [
            uv_path, "run",
            "--python", "3.10",
            "--directory", install_path,
            "python", "server.py"
        ]

    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "name": NetworkXServer.name,
            "description": NetworkXServer.description,
            "command": NetworkXServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
