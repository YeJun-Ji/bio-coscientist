"""
RFdiffusion MCP Server Configuration
Provides de novo protein design using RFdiffusion

RFdiffusion provides:
- De novo protein backbone generation
- Binder design (given target structure)
- Scaffold conditioning
- Partial diffusion for structure modification

REQUIRES GPU: CUDA-enabled GPU with 16GB+ memory (A100 40GB recommended)
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

from ..server_templates import install_server_template

logger = logging.getLogger(__name__)


class RFdiffusionServer:
    """
    RFdiffusion MCP Server - De Novo Protein Design

    Provides tools for:
    - design_binder: Design binder for a target protein
    - unconditional_design: Generate novel protein backbones
    - scaffold_conditioning: Design based on scaffold constraints

    Used for:
    - Problem 3: TNFR mini-binder backbone design
    - Problem 4: IL-11 inhibitor scaffold design

    REQUIRES: CUDA-enabled GPU with 16GB+ memory
    Model weights: ~2GB at ~/.biocoscientist/models/rfdiffusion/
    """

    name = "rfdiffusion"
    description = "RFdiffusion de novo protein design (GPU required)"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for RFdiffusion server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/rfdiffusion-mcp-server")

    @staticmethod
    def get_model_path() -> str:
        """Get the model weights path"""
        return os.path.expanduser("~/.biocoscientist/models/rfdiffusion")

    @staticmethod
    def is_installed() -> bool:
        """Check if RFdiffusion server is installed"""
        install_path = RFdiffusionServer.get_install_path()
        server_path = os.path.join(install_path, "server.py")
        venv_path = os.path.join(install_path, ".venv")
        return os.path.exists(server_path) and os.path.exists(venv_path)

    @staticmethod
    def install():
        """Install RFdiffusion MCP server from template"""
        install_path = RFdiffusionServer.get_install_path()
        model_path = RFdiffusionServer.get_model_path()

        if RFdiffusionServer.is_installed():
            logger.info(f"RFdiffusion server already installed at {install_path}")
            return

        logger.info(f"Installing RFdiffusion server to {install_path}")

        # Create directories
        os.makedirs(install_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        # Install server.py from template
        server_path = os.path.join(install_path, "server.py")
        if not os.path.exists(server_path):
            if not install_server_template("rfdiffusion", install_path):
                raise RuntimeError("Failed to install RFdiffusion server template")
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
            [uv_path, "pip", "install", "--python", venv_python, "mcp", "torch", "numpy", "hydra-core", "omegaconf", "biopython"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ RFdiffusion server setup complete")
        logger.info(f"Note: Download model weights to {model_path}")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start RFdiffusion server"""
        install_path = RFdiffusionServer.get_install_path()
        venv_python = os.path.join(install_path, ".venv", "bin", "python")
        server_path = os.path.join(install_path, "server.py")

        if not os.path.exists(server_path):
            raise RuntimeError(
                f"RFdiffusion server not found at {server_path}. "
                "Please run RFdiffusionServer.install() first."
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
            "name": RFdiffusionServer.name,
            "description": RFdiffusionServer.description,
            "command": RFdiffusionServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "evolution"],
            "category": "analysis",
            "requires_gpu": True,
            "min_gpu_memory_gb": 16,
        }
