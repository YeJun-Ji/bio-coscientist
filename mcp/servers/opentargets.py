"""
Open Targets MCP Server Configuration
Provides drug-target association and druggability analysis using Open Targets Platform API

Open Targets provides:
- Target-drug associations
- Druggability assessment
- Disease associations
- Drug information
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class OpenTargetsServer:
    """
    Open Targets MCP Server - Drug-Target Intelligence

    Provides tools for:
    - search_target_drugs: Find drugs targeting a specific gene/protein
    - assess_druggability: Evaluate target druggability
    - get_drug_info: Get drug details
    - get_target_info: Get target information

    Used for:
    - Problem 3: Evaluate TNFR druggability and existing drugs
    - Problem 4: IL-11 pathway druggability assessment
    """

    name = "opentargets"
    description = "Open Targets drug-target associations and druggability"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Open Targets server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/opentargets-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if Open Targets server is installed"""
        install_path = OpenTargetsServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install Open Targets MCP server from template"""
        install_path = OpenTargetsServer.get_install_path()

        if OpenTargetsServer.is_installed():
            logger.info(f"Open Targets server already installed at {install_path}")
            return

        logger.info(f"Installing Open Targets server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("opentargets", install_path):
                raise RuntimeError("Failed to install Open Targets server template")
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

        logger.info("✅ Open Targets server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Open Targets server"""
        install_path = OpenTargetsServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"Open Targets server not found at {server_path}. "
                "Please run OpenTargetsServer.install() first."
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
            "name": OpenTargetsServer.name,
            "description": OpenTargetsServer.description,
            "command": OpenTargetsServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",
        }
