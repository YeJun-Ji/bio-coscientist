"""
UniProt MCP Server Configuration
Provides access to UniProt protein database for sequence retrieval and protein information
"""

import os
import subprocess
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UniProtServer:
    """
    UniProt MCP Server - Protein Database Access

    Provides 26 tools for:
    - search_proteins: Search by protein name, keywords, or organism
    - get_protein_info: Retrieve comprehensive protein information
    - get_protein_sequence: Get amino acid sequences (FASTA/JSON)
    - search_by_gene: Find proteins by gene name
    - get_protein_features: Access functional domains, active sites, binding sites
    - And 21 more tools for comparative, structural, and batch analysis

    Key Use Case:
    - Convert protein name (e.g., "TNFR1") to amino acid sequence for ESMFold
    """

    name = "uniprot"
    description = "UniProt protein database (sequences, annotations, domains)"
    github_url = "https://github.com/Augmented-Nature/Augmented-Nature-UniProt-MCP-Server"

    @staticmethod
    def get_install_path() -> str:
        """Get the installation path for UniProt server"""
        return os.path.expanduser("~/.biocoscientist/mcp-servers/uniprot-mcp-server")

    @staticmethod
    def is_installed() -> bool:
        """Check if UniProt server is installed and built"""
        install_path = UniProtServer.get_install_path()
        build_path = os.path.join(install_path, "build", "index.js")
        return os.path.exists(build_path)

    @staticmethod
    def install():
        """Install UniProt MCP server from GitHub"""
        install_path = UniProtServer.get_install_path()

        if UniProtServer.is_installed():
            logger.info(f"UniProt server already installed at {install_path}")
            return

        logger.info("Installing UniProt MCP server...")

        # Create directory if needed
        os.makedirs(os.path.dirname(install_path), exist_ok=True)

        # Clone repository
        if not os.path.exists(install_path):
            logger.info(f"Cloning from {UniProtServer.github_url}...")
            subprocess.run(
                ["git", "clone", UniProtServer.github_url, install_path],
                check=True,
                capture_output=True
            )

        # Install npm dependencies
        logger.info("Installing npm dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        # Build TypeScript
        logger.info("Building TypeScript...")
        subprocess.run(
            ["npm", "run", "build"],
            cwd=install_path,
            check=True,
            capture_output=True
        )

        logger.info("✅ UniProt server installed successfully")

    @staticmethod
    def get_command() -> List[str]:
        """Get the command to start UniProt server"""
        install_path = UniProtServer.get_install_path()
        build_path = os.path.join(install_path, "build", "index.js")

        if not os.path.exists(build_path):
            raise RuntimeError(
                f"UniProt server not found at {build_path}. "
                "Please run UniProtServer.install() first."
            )

        return ["node", build_path]

    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "name": UniProtServer.name,
            "description": UniProtServer.description,
            "command": UniProtServer.get_command(),
            "args": [],
            "auto_install": True,
            "problem_types": ["all"],
            "stages": ["generation", "reflection", "evolution"],
            "category": "collection",
        }
