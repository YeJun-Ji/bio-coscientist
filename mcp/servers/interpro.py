"""
InterPro MCP Server Configuration
Provides protein domain and family analysis using InterPro API

InterPro provides:
- Domain architecture analysis
- Protein family classification
- Functional site prediction
- Cross-database integration (Pfam, PROSITE, SMART, etc.)
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class InterProServer:
    """
    InterPro MCP Server - Protein Domain Analysis

    Provides tools for:
    - analyze_domains: Analyze protein domains from sequence or UniProt ID
    - get_entry_info: Get InterPro entry details
    - get_domain_architecture: Get visual domain architecture
    - search_by_domain: Find proteins with specific domain
    - compare_domain_architectures: Compare multiple proteins

    Used for:
    - Problem 3: TNFR1/2 domain analysis for binder design
    - Problem 4: IL-11 and interactor functional domain analysis
    """

    name = "interpro"
    description = "InterPro protein domain and family analysis"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for InterPro server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/interpro-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if InterPro server is installed"""
        install_path = InterProServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install InterPro MCP server"""
        install_path = InterProServer.get_install_path()

        if InterProServer.is_installed():
            logger.info(f"InterPro server already installed at {install_path}")
            return

        logger.info(f"Installing InterPro server at {install_path}")

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

        logger.info("✅ InterPro server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start InterPro server"""
        install_path = InterProServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"InterPro server not found at {server_path}. "
                "Please run InterProServer.install() first."
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
            "name": InterProServer.name,
            "description": InterProServer.description,
            "command": InterProServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
        }
