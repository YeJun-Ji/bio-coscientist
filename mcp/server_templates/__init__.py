"""
MCP Server Templates

This module contains server.py templates for all MCP servers.
These templates are copied to ~/.biocoscientist/mcp-servers/ during installation.
"""

import os
import shutil
import logging

logger = logging.getLogger(__name__)

# Directory containing this module (server_templates/)
TEMPLATES_DIR = os.path.dirname(os.path.abspath(__file__))


def get_template_path(server_name: str) -> str:
    """
    Get the path to a server template file.

    Args:
        server_name: Name of the server (e.g., 'esmfold', 'blast', 'stringdb')

    Returns:
        Full path to the template file
    """
    template_file = f"{server_name}_server.py"
    return os.path.join(TEMPLATES_DIR, template_file)


def install_server_template(server_name: str, install_path: str) -> bool:
    """
    Install a server.py template to the target installation directory.

    Args:
        server_name: Name of the server (e.g., 'esmfold', 'blast')
        install_path: Target directory (e.g., ~/.biocoscientist/mcp-servers/esmfold-mcp-server/)

    Returns:
        True if successful, False otherwise
    """
    template_path = get_template_path(server_name)
    dest_path = os.path.join(install_path, "server.py")

    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_path}")
        return False

    try:
        # Create install directory if needed
        os.makedirs(install_path, exist_ok=True)

        # Copy template to destination
        shutil.copy(template_path, dest_path)
        logger.info(f"Installed server template: {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to install template: {e}")
        return False


def list_available_templates() -> list:
    """List all available server templates."""
    templates = []
    for f in os.listdir(TEMPLATES_DIR):
        if f.endswith("_server.py"):
            name = f.replace("_server.py", "")
            templates.append(name)
    return sorted(templates)
