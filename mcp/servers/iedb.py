"""
IEDB MCP Server Configuration
Provides MHC binding prediction using IEDB API (alternative to NetMHCpan)

IEDB provides:
- MHC-I binding prediction
- MHC-II binding prediction
- Epitope prediction
- Immunogenicity scoring
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class IEDBServer:
    """
    IEDB MCP Server - Immunogenicity Prediction

    Provides tools for:
    - predict_mhc_binding: MHC-I binding prediction
    - predict_mhc_ii_binding: MHC-II binding prediction
    - scan_protein: Scan protein for epitopes
    - list_alleles: Get supported HLA alleles

    Used for:
    - Problem 3: Binder immunogenicity assessment
    - Problem 4: IL-11 binder safety evaluation
    """

    name = "iedb"
    description = "IEDB MHC binding and immunogenicity prediction"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for IEDB server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/iedb-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if IEDB server is installed"""
        install_path = IEDBServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install IEDB MCP server"""
        install_path = IEDBServer.get_install_path()

        if IEDBServer.is_installed():
            logger.info(f"IEDB server already installed at {install_path}")
            return

        logger.info(f"Installing IEDB server at {install_path}")

        os.makedirs(install_path, exist_ok=True)

        requirements_path = os.path.join(install_path, "requirements.txt")
        if not os.path.exists(requirements_path):
            with open(requirements_path, "w") as f:
                f.write("requests>=2.28.0\n")
                f.write("pandas>=2.0.0\n")
                f.write("mcp>=0.1.0\n")

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

            uv_path = os.path.expanduser("~/.local/bin/uv")
        venv_python = os.path.join(venv_path, "bin", "python")
        logger.info("Installing dependencies...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "-r", requirements_path],
                cwd=install_path,
                check=True,
                capture_output=True
            )

        logger.info("✅ IEDB server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start IEDB server"""
        install_path = IEDBServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"IEDB server not found at {server_path}. "
                "Please run IEDBServer.install() first."
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
            "name": IEDBServer.name,
            "description": IEDBServer.description,
            "command": IEDBServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
