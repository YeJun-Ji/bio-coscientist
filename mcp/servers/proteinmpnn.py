"""
ProteinMPNN MCP Server Configuration
Provides structure-based protein sequence design using ProteinMPNN

ProteinMPNN provides:
- Inverse folding (backbone → sequence)
- Fixed position design
- Multi-state design
- Sequence scoring

REQUIRES GPU: CUDA-enabled GPU with 8GB+ memory
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ProteinMPNNServer:
    """
    ProteinMPNN MCP Server - Structure-Based Sequence Design

    Provides tools for:
    - design_sequence: Design sequence for a given backbone
    - design_with_fixed_positions: Design with some positions fixed
    - score_sequence: Score sequence-structure compatibility

    Used for:
    - Problem 3: TNFR binder sequence design
    - Problem 4: IL-11 inhibitor sequence optimization

    REQUIRES: CUDA-enabled GPU with 8GB+ memory
    Model weights: ~150MB at ~/.biocoscientist/models/proteinmpnn/
    """

    name = "proteinmpnn"
    description = "ProteinMPNN structure-based sequence design (GPU required)"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for ProteinMPNN server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/proteinmpnn-mcp-server")

    @staticmethod
    def get_model_path() -> str:
        """Get the model weights path"""
        return os.path.expanduser("~/.biocoscientist/models/proteinmpnn")

    @staticmethod
    def is_installed() -> bool:
        """Check if ProteinMPNN server is installed"""
        install_path = ProteinMPNNServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install ProteinMPNN MCP server"""
        install_path = ProteinMPNNServer.get_install_path()
        model_path = ProteinMPNNServer.get_model_path()

        if ProteinMPNNServer.is_installed():
            logger.info(f"ProteinMPNN server already installed at {install_path}")
            return

        logger.info(f"Installing ProteinMPNN server at {install_path}")

        os.makedirs(install_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        requirements_path = os.path.join(install_path, "requirements.txt")
        if not os.path.exists(requirements_path):
            with open(requirements_path, "w") as f:
                f.write("torch>=2.0.0\n")
                f.write("numpy>=1.24.0\n")
                f.write("biopython>=1.81\n")
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

        logger.info("✅ ProteinMPNN server setup complete")
        logger.info(f"Note: Download model weights to {model_path}")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start ProteinMPNN server"""
        install_path = ProteinMPNNServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"ProteinMPNN server not found at {server_path}. "
                "Please run ProteinMPNNServer.install() first."
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
            "name": ProteinMPNNServer.name,
            "description": ProteinMPNNServer.description,
            "command": ProteinMPNNServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "evolution"],
            "category": "analysis",
            "requires_gpu": True,
            "min_gpu_memory_gb": 8,
        }
