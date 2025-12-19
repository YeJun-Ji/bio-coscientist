"""
MCP Server Configurations
Each module defines how to connect to and configure a specific MCP server
"""

from .kegg import KEGGServer
from .rosetta import RosettaServer
from .reactome import ReactomeServer
from .ncbi import NCBIServer
from .pymol import PyMolServer
from .scholar import ScholarServer
from .esmfold import ESMFoldServer
from .uniprot import UniProtServer
from .blast import BLASTServer

__all__ = [
    "KEGGServer",
    "RosettaServer",
    "ReactomeServer",
    "NCBIServer",
    "PyMolServer",
    "ScholarServer",
    "ESMFoldServer",
    "UniProtServer",
    "BLASTServer",
]
