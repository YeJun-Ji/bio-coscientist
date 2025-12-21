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
        """Install gProfiler MCP server"""
        install_path = GProfilerServer.get_install_path()

        if GProfilerServer.is_installed():
            logger.info(f"gProfiler server already installed at {install_path}")
            return

        logger.info(f"Installing gProfiler server at {install_path}")

        os.makedirs(install_path, exist_ok=True)

        requirements_path = os.path.join(install_path, "requirements.txt")
        if not os.path.exists(requirements_path):
            with open(requirements_path, "w") as f:
                f.write("gprofiler-official>=1.0.0\n")
                f.write("requests>=2.28.0\n")
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
            "category": "analysis",
        }
