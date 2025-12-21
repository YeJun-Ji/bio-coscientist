"""
STRING DB MCP Server Configuration
Provides protein-protein interaction network analysis using STRING database API

STRING DB provides:
- Protein-protein interaction networks
- Interaction confidence scores (experimental, database, textmining, etc.)
- Functional enrichment analysis (GO terms, KEGG pathways)
- Network visualization
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class STRINGDBServer:
    """
    STRING DB MCP Server - Protein-Protein Interaction Networks

    Provides tools for:
    - get_protein_network: Get PPI network for a protein
    - get_interaction_partners: Get interaction partners with scores
    - get_enrichment_analysis: Functional enrichment (GO, KEGG)
    - get_network_image: Network visualization URL

    Confidence Score Interpretation (0-1000):
    - >= 900: Highest confidence
    - >= 700: High confidence
    - >= 400: Medium confidence (default)
    - >= 150: Low confidence

    Used primarily for:
    - Problem 4: IL-11 interactor discovery
    - Problem 5: Drug-gene network analysis
    """

    name = "stringdb"
    description = "STRING DB protein-protein interaction network analysis"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for STRING DB server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/stringdb-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if STRING DB server is installed"""
        install_path = STRINGDBServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install STRING DB MCP server"""
        install_path = STRINGDBServer.get_install_path()

        if STRINGDBServer.is_installed():
            logger.info(f"STRING DB server already installed at {install_path}")
            return

        logger.info(f"Installing STRING DB server at {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Create requirements.txt
        requirements_path = os.path.join(install_path, "requirements.txt")
        if not os.path.exists(requirements_path):
            with open(requirements_path, "w") as f:
                f.write("requests>=2.28.0\n")
                f.write("mcp>=0.1.0\n")

        # Check if venv exists, if not create it
        venv_path = os.path.join(install_path, ".venv")
        venv_python = os.path.join(venv_path, "bin", "python")
        uv_path = os.path.expanduser("~/.local/bin/uv")

        if not os.path.exists(venv_python):
            logger.info("Creating virtual environment...")

            # Create venv using uv
            subprocess.run(
                [uv_path, "venv", "--python", "3.10", ".venv"],
                cwd=install_path,
                check=True,
                capture_output=True
            )

        # Install requirements using uv pip (not venv pip)
        logger.info("Installing dependencies...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "-r", requirements_path],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ STRING DB server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start STRING DB server"""
        install_path = STRINGDBServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"STRING DB server not found at {server_path}. "
                "Please run STRINGDBServer.install() first."
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
            "name": STRINGDBServer.name,
            "description": STRINGDBServer.description,
            "command": STRINGDBServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],  # P4, P5 primarily
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
