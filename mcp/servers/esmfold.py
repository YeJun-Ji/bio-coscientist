"""
ESMFold MCP Server Configuration
Provides protein structure prediction using ESMFold API with pLDDT confidence scores
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ESMFoldServer:
    """
    ESMFold MCP Server - Protein Structure Prediction

    Provides tools for:
    - fold_sequence: Predict 3D structure from protein sequence
    - get_plddt_score: Get pLDDT confidence score
    - validate_structure: Comprehensive structure quality validation

    pLDDT Score Interpretation:
    - >= 90: Excellent - very high confidence
    - 70-90: Good - confident prediction
    - 50-70: Moderate - use with caution
    - < 50: Poor - likely disordered region
    """

    name = "esmfold"
    description = "ESMFold protein structure prediction (pLDDT confidence scores)"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for ESMFold server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/esmfold-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if ESMFold server is installed"""
        install_path = ESMFoldServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install ESMFold MCP server (already created, just verify)"""
        install_path = ESMFoldServer.get_install_path()

        if ESMFoldServer.is_installed():
            logger.info(f"ESMFold server already installed at {install_path}")
            return

        logger.info("ESMFold server not found. Please run the setup manually.")
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

        logger.info("✅ ESMFold server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start ESMFold server"""
        install_path = ESMFoldServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"ESMFold server not found at {server_path}. "
                "Please run ESMFoldServer.install() first."
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
            "name": ESMFoldServer.name,
            "description": ESMFoldServer.description,
            "command": ESMFoldServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
