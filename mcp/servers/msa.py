"""
MSA (Multiple Sequence Alignment) MCP Server Configuration
Provides multiple sequence alignment and phylogenetic analysis using Clustal Omega web service

MSA analysis provides:
- Multiple sequence alignment (Clustal Omega via EMBL-EBI)
- Sequence conservation scoring
- Consensus sequence generation
- Pairwise identity matrix
- Simple distance-based phylogenetic tree (UPGMA)
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class MSAServer:
    """
    MSA MCP Server - Multiple Sequence Alignment and Phylogenetics

    Provides tools for:
    - align_sequences: Multiple sequence alignment (Clustal Omega)
    - calculate_conservation: Sequence conservation scores
    - get_consensus_sequence: Consensus sequence generation
    - get_pairwise_identity: Pairwise identity matrix
    - build_distance_tree: Simple UPGMA phylogenetic tree

    Conservation Score Interpretation:
    - 1.0 (100%): Fully conserved position
    - >= 0.9: Highly conserved
    - >= 0.7: Moderately conserved
    - < 0.5: Variable position

    Uses EMBL-EBI Clustal Omega web service:
    - No local installation required
    - Supports protein and DNA sequences
    - Rate limited (be mindful of large batches)

    Used primarily for:
    - Problem 1: T-cell gene functional similarity via phylogenetic analysis
    """

    name = "msa"
    description = "Multiple sequence alignment and phylogenetic analysis (Clustal Omega)"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for MSA server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/msa-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if MSA server is installed"""
        install_path = MSAServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install MSA MCP server from template"""
        install_path = MSAServer.get_install_path()

        if MSAServer.is_installed():
            logger.info(f"MSA server already installed at {install_path}")
            return

        logger.info(f"Installing MSA server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("msa", install_path):
                raise RuntimeError("Failed to install MSA server template")
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
            [uv_path, "pip", "install", "--python", venv_python, "mcp", "requests"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ MSA server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start MSA server"""
        install_path = MSAServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"MSA server not found at {server_path}. "
                "Please run MSAServer.install() first."
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
            "name": MSAServer.name,
            "description": MSAServer.description,
            "command": MSAServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],  # P1 primarily
            "stages": ["generation", "analysis"],
            "category": "analysis",
        }
