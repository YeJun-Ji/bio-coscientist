"""
gProfiler MCP Server Configuration
Provides GO/Pathway enrichment analysis using g:Profiler API

g:Profiler provides:
- GO enrichment analysis (BP, MF, CC)
- Pathway analysis (KEGG, Reactome)
- Gene ID conversion
- Multi-query comparison
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class GProfilerServer:
    """
    gProfiler MCP Server - Functional Enrichment Analysis

    Provides tools for:
    - enrichment_analysis: GO/KEGG/Reactome enrichment
    - convert_gene_ids: Gene ID conversion between databases
    - compare_gene_lists: Compare multiple gene lists

    Used for:
    - Problem 4: IL-11 interactor functional analysis
    - Pathway and GO term enrichment for target validation
    """

    name = "gprofiler"
    description = "g:Profiler functional enrichment analysis (GO, KEGG, Reactome)"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for gProfiler server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/gprofiler-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if gProfiler server is installed"""
        install_path = GProfilerServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install gProfiler MCP server from template"""
        install_path = GProfilerServer.get_install_path()

        if GProfilerServer.is_installed():
            logger.info(f"gProfiler server already installed at {install_path}")
            return

        logger.info(f"Installing gProfiler server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("gprofiler", install_path):
                raise RuntimeError("Failed to install gProfiler server template")
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
            [uv_path, "pip", "install", "--python", venv_python, "mcp", "requests", "gprofiler-official"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ gProfiler server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start gProfiler server"""
        install_path = GProfilerServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"gProfiler server not found at {server_path}. "
                "Please run GProfilerServer.install() first."
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
            "name": GProfilerServer.name,
            "description": GProfilerServer.description,
            "command": GProfilerServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",  # Queries external g:Profiler database
        }
