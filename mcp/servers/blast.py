"""
BLAST MCP Server Configuration
Provides sequence similarity search using NCBI BLAST API
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BLASTServer:
    """
    BLAST MCP Server - Sequence Similarity Search

    Provides tools for:
    - blastp_search: Submit protein vs protein BLAST search
    - check_blast_status: Check job status using RID
    - get_blast_results: Retrieve completed results
    - find_similar_proteins: High-level integrated search

    Use Cases:
    - Find homologous proteins for evolutionary analysis
    - Off-target prediction for designed proteins
    - Validate novelty of designed sequences

    API Limitations:
    - Max 1 request per 10 seconds
    - Max 1 status check per minute per RID
    - RIDs expire after 24 hours
    """

    name = "blast"
    description = "NCBI BLAST sequence similarity search"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for BLAST server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/blast-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if BLAST server is installed"""
        install_path = BLASTServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install BLAST MCP server (verify setup)"""
        install_path = BLASTServer.get_install_path()

        if BLASTServer.is_installed():
            logger.info(f"BLAST server already installed at {install_path}")
            return

        logger.info("BLAST server not found. Please run the setup manually.")
        logger.info(f"Expected location: {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Check if venv exists, if not create it
        venv_path = os.path.join(install_path, ".venv")
        if not os.path.exists(venv_path):
            logger.info("Creating virtual environment...")
            uv_path = os.path.expanduser("~/.local/bin/uv")
            subprocess.run(
                [uv_path, "venv", "--python", "3.10", ".venv"],
                cwd=install_path,
                check=True,
                capture_output=True
            )

            # Install requirements
            requirements_path = os.path.join(install_path, "requirements.txt")
            if os.path.exists(requirements_path):
                logger.info("Installing dependencies...")
                subprocess.run(
                    [uv_path, "pip", "install", "-r", requirements_path],
                    cwd=install_path,
                    check=True,
                    capture_output=True
                )

        logger.info("✅ BLAST server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start BLAST server"""
        install_path = BLASTServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"BLAST server not found at {server_path}. "
                "Please run BLASTServer.install() first."
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
            "name": BLASTServer.name,
            "description": BLASTServer.description,
            "command": BLASTServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
