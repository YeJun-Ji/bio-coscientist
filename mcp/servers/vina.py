"""
AutoDock Vina MCP Server Configuration
Provides molecular docking and binding affinity prediction

Vina provides:
- Protein-ligand docking
- Binding affinity estimation
- Receptor/ligand preparation
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class VinaServer:
    """
    AutoDock Vina MCP Server - Molecular Docking

    Provides tools for:
    - dock_ligand: Dock ligand to receptor
    - prepare_receptor: Prepare receptor PDBQT
    - prepare_ligand: Prepare ligand from SMILES
    - calculate_binding_affinity: Get binding scores

    Used for:
    - Problem 3: TNFR binder affinity prediction
    - Problem 4: IL-11 inhibitor docking analysis

    Note: Requires AutoDock Vina binary installed on system
    """

    name = "vina"
    description = "AutoDock Vina molecular docking and binding affinity"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Vina server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/vina-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if Vina server is installed"""
        install_path = VinaServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install Vina MCP server from template"""
        install_path = VinaServer.get_install_path()

        if VinaServer.is_installed():
            logger.info(f"Vina server already installed at {install_path}")
            return

        logger.info(f"Installing Vina server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("vina", install_path):
                raise RuntimeError("Failed to install Vina server template")
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

        # Install dependencies (note: vina/meeko require Boost C++ library)
        logger.info("Installing dependencies...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "mcp", "requests", "biopython"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ Vina server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Vina server"""
        install_path = VinaServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"Vina server not found at {server_path}. "
                "Please run VinaServer.install() first."
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
            "name": VinaServer.name,
            "description": VinaServer.description,
            "command": VinaServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
