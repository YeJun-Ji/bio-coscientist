"""
RCSB PDB MCP Server Configuration
Provides protein structure download and binding site analysis using RCSB PDB API

RCSB PDB provides:
- Structure download (PDB/mmCIF format)
- Structure search by name, sequence, or other criteria
- Binding site analysis
- Ligand interaction data
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class RCSBPDBServer:
    """
    RCSB PDB MCP Server - Protein Structure Database

    Provides tools for:
    - download_pdb: Download structure by PDB ID
    - search_structures: Search structures by name/keyword
    - get_binding_sites: Analyze binding sites in a structure
    - get_ligand_interactions: Get ligand-protein interactions
    - search_by_sequence: Search by sequence similarity
    - get_protein_info: Get structure metadata

    Used for:
    - Problem 3: TNFR1/2 structure analysis and binding site selection
    - Problem 4: IL-11 and interactor structure analysis
    """

    name = "rcsbpdb"
    description = "RCSB PDB structure database - download, search, binding site analysis"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for RCSB PDB server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/rcsbpdb-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if RCSB PDB server is installed"""
        install_path = RCSBPDBServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install RCSB PDB MCP server from template"""
        install_path = RCSBPDBServer.get_install_path()

        if RCSBPDBServer.is_installed():
            logger.info(f"RCSB PDB server already installed at {install_path}")
            return

        logger.info(f"Installing RCSB PDB server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("rcsbpdb", install_path):
                raise RuntimeError("Failed to install RCSB PDB server template")
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
            [uv_path, "pip", "install", "--python", venv_python, "mcp", "requests", "biopython"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ RCSB PDB server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start RCSB PDB server"""
        install_path = RCSBPDBServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"RCSB PDB server not found at {server_path}. "
                "Please run RCSBPDBServer.install() first."
            )

        # Use venv Python directly
        if os.path.exists(venv_python):
            return [venv_python, server_path]

        # Fallback to uv run
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
            "name": RCSBPDBServer.name,
            "description": RCSBPDBServer.description,
            "command": RCSBPDBServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",
        }
