"""
Rosetta Integration - PyRosetta direct usage for protein design
Note: This is NOT an MCP server integration. It uses PyRosetta library directly.
"""

from .rosetta.rosetta_client import RosettaClient
from .rosetta.rosetta_task_manager import RosettaTaskManager

__all__ = ["RosettaClient", "RosettaTaskManager"]
