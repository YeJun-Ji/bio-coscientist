"""
ChEMBL MCP Server Configuration
Provides drug-target interaction data and compound information from ChEMBL database

ChEMBL provides:
- Drug-target binding affinity data (IC50, Ki, Kd)
- Approved drug information
- Mechanism of action
- Compound bioactivity profiles
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ChEMBLServer:
    """
    ChEMBL MCP Server - Drug-Target Interaction Database

    Provides tools for:
    - search_target: Search for drug targets by name
    - get_target_info: Get detailed target information
    - search_compound: Search for compounds/drugs
    - get_compound_activities: Get bioactivity data (IC50, Ki, Kd)
    - get_drug_mechanisms: Get mechanism of action
    - search_approved_drugs: Find approved drugs for a target
    - get_target_bioactivities: Get all compounds tested against a target

    Activity Value Interpretation (pChEMBL):
    - >= 8: High potency (< 10 nM)
    - 6-8: Moderate potency (10-1000 nM)
    - < 6: Low potency (> 1000 nM)

    Max Phase:
    - 4: Approved drug
    - 3: Phase III clinical trial
    - 2: Phase II clinical trial
    - 1: Phase I clinical trial
    - 0: Preclinical

    Used primarily for:
    - Problem 5: T-cell exhaustion drug repurposing
    """

    name = "chembl"
    description = "ChEMBL drug-target interaction database"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for ChEMBL server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/chembl-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if ChEMBL server is installed"""
        install_path = ChEMBLServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install ChEMBL MCP server"""
        install_path = ChEMBLServer.get_install_path()

        if ChEMBLServer.is_installed():
            logger.info(f"ChEMBL server already installed at {install_path}")
            return

        logger.info(f"Installing ChEMBL server at {install_path}")

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
        if not os.path.exists(venv_path):
            logger.info("Creating virtual environment...")
            uv_path = os.path.expanduser("~/.local/bin/uv")

            # Create venv
            subprocess.run(
                [uv_path, "venv", "--python", "3.10", ".venv"],
                cwd=install_path,
                check=True,
                capture_output=True
            )

            # Install requirements
            uv_path = os.path.expanduser("~/.local/bin/uv")
        venv_python = os.path.join(venv_path, "bin", "python")
        logger.info("Installing dependencies...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "-r", requirements_path],
                cwd=install_path,
                check=True,
                capture_output=True
            )

        logger.info("✅ ChEMBL server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start ChEMBL server"""
        install_path = ChEMBLServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"ChEMBL server not found at {server_path}. "
                "Please run ChEMBLServer.install() first."
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
            "name": ChEMBLServer.name,
            "description": ChEMBLServer.description,
            "command": ChEMBLServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],  # P5 primarily
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
