"""
Nanopore MCP Server Configuration
Provides Oxford Nanopore sequencing data analysis for poly(A) tail and modified base detection

Nanopore analysis provides:
- poly(A) tail length distribution analysis
- Modified base detection (5mC, 6mA)
- BAM file alignment statistics
- Distribution comparison between samples
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class NanoporeServer:
    """
    Nanopore MCP Server - Oxford Nanopore Sequencing Analysis

    Provides tools for:
    - analyze_polya_lengths: Poly(A) tail length distribution
    - get_alignment_stats: BAM alignment statistics
    - detect_modified_bases: Modified base detection (5mC, 6mA)
    - compare_polya_distributions: Statistical comparison (KS test)
    - extract_polya_region: Region-specific poly(A) analysis
    - analyze_non_a_bases: Non-A base detection in poly(A) tails

    Poly(A) Length Interpretation:
    - Short (<50 nt): May indicate degradation or deadenylation
    - Medium (50-150 nt): Typical for stable mRNAs
    - Long (>150 nt): Newly synthesized or stabilized transcripts

    Requirements:
    - pysam: BAM file processing
    - numpy/scipy: Statistical analysis
    - POD5 files need dorado basecaller for poly(A) annotations

    Used primarily for:
    - Problem 2: RNA virus poly(A) tail analysis
    """

    name = "nanopore"
    description = "Nanopore sequencing poly(A) and modified base analysis"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for Nanopore server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/nanopore-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if Nanopore server is installed"""
        install_path = NanoporeServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install Nanopore MCP server from template"""
        install_path = NanoporeServer.get_install_path()

        if NanoporeServer.is_installed():
            logger.info(f"Nanopore server already installed at {install_path}")
            return

        logger.info(f"Installing Nanopore server to {install_path}")

        # Create directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("nanopore", install_path):
                raise RuntimeError("Failed to install Nanopore server template")
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
        logger.info("Installing dependencies (pysam, numpy, scipy)...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "mcp", "pysam", "numpy", "scipy"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ Nanopore server setup complete")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start Nanopore server"""
        install_path = NanoporeServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"Nanopore server not found at {server_path}. "
                "Please run NanoporeServer.install() first."
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
            "name": NanoporeServer.name,
            "description": NanoporeServer.description,
            "command": NanoporeServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],  # P2 primarily
            "stages": ["generation", "analysis"],
            "category": "analysis",
        }
