"""
Pandas MCP Server Configuration
Provides pandas-based data analysis, statistical testing, and visualization
"""

import os
import subprocess
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PandasAnalysisServer:
    """
    Pandas Data Analysis MCP Server
    GitHub: https://github.com/marlonluo2018/pandas-mcp-server

    Provides tools for:
    - Data loading and preprocessing
    - Statistical analysis (correlation, t-test, ANOVA)
    - Data transformation (filtering, grouping, aggregation)
    - Clustering and dimensionality reduction
    - Data visualization
    """

    name = "pandas_analysis"
    description = "Pandas-based data analysis, statistics, and visualization"
    github_url = "https://github.com/marlonluo2018/pandas-mcp-server"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Pandas server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/pandas-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if Pandas server is installed with venv"""
        install_path = PandasAnalysisServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install Pandas MCP server from GitHub with venv"""
        install_path = PandasAnalysisServer.get_install_path()
        uv_path = os.path.expanduser("~/.local/bin/uv")

        # Clone if not already cloned
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            logger.info("Installing Pandas MCP server...")
            os.makedirs(os.path.dirname(install_path), exist_ok=True)

            logger.info(f"Cloning from {PandasAnalysisServer.github_url}...")
            subprocess.run(
                ["git", "clone", PandasAnalysisServer.github_url, install_path],
                check=True,
                capture_output=True
            )

        # Create venv if needed
        venv_path = os.path.join(install_path, ".venv")
        venv_python = os.path.join(venv_path, "bin", "python")

        if not os.path.exists(venv_python):
            logger.info("Creating virtual environment...")
            subprocess.run(
                [uv_path, "venv", "--python", "3.10", ".venv"],
                cwd=install_path,
                check=True,
                capture_output=True
            )

        # Install dependencies
        requirements_path = os.path.join(install_path, "requirements.txt")
        if os.path.exists(requirements_path):
            logger.info("Installing dependencies...")
            subprocess.run(
                [uv_path, "pip", "install", "--python", venv_python, "-r", requirements_path],
                cwd=install_path,
                check=True,
                capture_output=True
            )

        # Install mcp package (required for FastMCP)
        logger.info("Installing mcp package...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "mcp>=1.0.0"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ Pandas server installed successfully")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Pandas server"""
        install_path = PandasAnalysisServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"Pandas server not found at {server_path}. "
                "Please run PandasAnalysisServer.install() first."
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
            "name": PandasAnalysisServer.name,
            "description": PandasAnalysisServer.description,
            "command": PandasAnalysisServer.get_command(),
            "args": [],
            "auto_install": True,  # Enable auto-install with venv
            "problem_types": ["all"],  # Useful for all data analysis problems
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",  # Data analysis category
            "cwd": PandasAnalysisServer.get_install_path(),  # Required for local module imports
        }
