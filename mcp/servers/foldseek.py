"""
Foldseek MCP Server Configuration
Provides protein structure similarity search using Foldseek web API

Foldseek enables fast and sensitive comparison of protein structures:
- Structure similarity search against AlphaFold DB and PDB
- TM-score based structural alignment
- 3Di structural alphabet for fast searching
- Pairwise structure comparison

Reference: van Kempen et al. "Fast and accurate protein structure search with Foldseek"
Nature Biotechnology (2024) https://doi.org/10.1038/s41587-023-01773-0
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class FoldseekServer:
    """
    Foldseek MCP Server - Protein Structure Similarity Search

    Provides tools for:
    - search_structure: Search structure against AlphaFold DB/PDB
    - compare_structures: Compare two structures (TM-score)
    - get_structural_neighbors: Find structurally similar proteins
    - calculate_structure_similarity_matrix: Pairwise similarity for gene set

    TM-score Interpretation:
    - >= 0.5: Same fold (significant structural similarity)
    - >= 0.7: High similarity (likely same superfamily)
    - >= 0.9: Very high similarity (nearly identical structures)
    - < 0.3: Random structural similarity

    Used primarily for:
    - Problem 1 Requirement 5: Protein structure similarity for top 300 genes

    API: Uses search.foldseek.com REST API (same backend as MMseqs2)
    """

    name = "foldseek"
    description = "Foldseek protein structure similarity search (TM-score, 3Di)"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Foldseek server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/foldseek-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if Foldseek server is installed"""
        install_path = FoldseekServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install Foldseek MCP server from template"""
        install_path = FoldseekServer.get_install_path()

        if FoldseekServer.is_installed():
            logger.info(f"Foldseek server already installed at {install_path}")
            return

        logger.info(f"Installing Foldseek server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("foldseek", install_path):
                raise RuntimeError("Failed to install Foldseek server template")
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

        logger.info("✅ Foldseek server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Foldseek server"""
        install_path = FoldseekServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"Foldseek server not found at {server_path}. "
                "Please run FoldseekServer.install() first."
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
            "name": FoldseekServer.name,
            "description": FoldseekServer.description,
            "command": FoldseekServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],  # Primarily P1 Req.5
            "stages": ["generation", "analysis"],
            "category": "analysis",
        }
