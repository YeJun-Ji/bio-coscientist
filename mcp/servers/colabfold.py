"""
ColabFold MCP Server Configuration
Provides protein structure prediction using ColabFold (AlphaFold2 + MMseqs2)

ColabFold provides:
- Single protein structure prediction
- Protein complex structure prediction
- Fast MSA generation using MMseqs2
- Confidence metrics (pLDDT, pTM, ipTM)

REQUIRES GPU: CUDA-enabled GPU with 24GB+ memory (A100 recommended)
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ColabFoldServer:
    """
    ColabFold MCP Server - Protein Structure Prediction

    Provides tools for:
    - predict_structure: Single protein structure prediction
    - predict_complex: Multi-protein complex prediction
    - predict_binder_complex: Binder-target complex prediction
    - get_confidence_metrics: pLDDT, pTM, ipTM scores

    Used for:
    - Problem 3: TNFR-binder complex structure prediction
    - Problem 4: IL-11 complex structure validation

    REQUIRES: CUDA-enabled GPU with 24GB+ memory
    Model weights: ~5GB at ~/.biocoscientist/models/colabfold/
    """

    name = "colabfold"
    description = "ColabFold protein structure prediction (GPU required)"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for ColabFold server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/colabfold-mcp-server")

    @staticmethod
    def get_model_path() -> str:
        """Get the model weights path"""
        return os.path.expanduser("~/.biocoscientist/models/colabfold")

    @staticmethod
    def is_installed() -> bool:
        """Check if ColabFold server is installed"""
        install_path = ColabFoldServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install ColabFold MCP server"""
        install_path = ColabFoldServer.get_install_path()
        model_path = ColabFoldServer.get_model_path()

        if ColabFoldServer.is_installed():
            logger.info(f"ColabFold server already installed at {install_path}")
            return

        logger.info(f"Installing ColabFold server at {install_path}")

        os.makedirs(install_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        requirements_path = os.path.join(install_path, "requirements.txt")
        if not os.path.exists(requirements_path):
            with open(requirements_path, "w") as f:
                # ColabFold with JAX CUDA support
                f.write("colabfold[alphafold]>=1.5.0\n")
                f.write("jax>=0.4.0\n")
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
        logger.info("Installing dependencies (this may take a while)...")
        subprocess.run(
            [uv_path, "pip", "install", "--python", venv_python, "-r", requirements_path],
                cwd=install_path,
                check=True,
                capture_output=True
            )

        logger.info("✅ ColabFold server setup complete")
        logger.info(f"Note: Download model weights to {model_path}")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start ColabFold server"""
        install_path = ColabFoldServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"ColabFold server not found at {server_path}. "
                "Please run ColabFoldServer.install() first."
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
            "name": ColabFoldServer.name,
            "description": ColabFoldServer.description,
            "command": ColabFoldServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "analysis",
            "requires_gpu": True,
            "min_gpu_memory_gb": 24,
        }
