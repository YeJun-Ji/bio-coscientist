"""
MCP Server Configurations
Each module defines how to connect to and configure a specific MCP server

Server Status (as of 2024-12):
- ACTIVE (7 original): KEGG, Rosetta, NCBI, ESMFold, UniProt, BLAST, PandasAnalysis
- ADDED (5 v1): STRING DB, ChEMBL, Nanopore, MSA, Foldseek
- ADDED (10 v2): RCSB PDB, InterPro, gProfiler, OpenTargets, NetworkX, IEDB, Vina,
                  ProteinMPNN (GPU), RFdiffusion (GPU), ColabFold (GPU)
- REMOVED (3): Reactome (→ KEGG), Scholar (→ NCBI), PyMol (local tool)
"""

# Original servers (kept)
from .kegg import KEGGServer
from .rosetta import RosettaServer
from .ncbi import NCBIServer
from .esmfold import ESMFoldServer
from .uniprot import UniProtServer
from .blast import BLASTServer
from .pandas_analysis import PandasAnalysisServer

# New servers - v1 (added)
from .stringdb import STRINGDBServer
from .chembl import ChEMBLServer
from .nanopore import NanoporeServer
from .msa import MSAServer
from .foldseek import FoldseekServer

# New servers - v2 (Problem 3 & 4 support)
# API/CPU servers
from .rcsbpdb import RCSBPDBServer
from .interpro import InterProServer
from .gprofiler import GProfilerServer
from .opentargets import OpenTargetsServer
from .networkx_server import NetworkXServer
from .iedb import IEDBServer
from .vina import VinaServer
# GPU servers
from .proteinmpnn import ProteinMPNNServer
from .rfdiffusion import RFdiffusionServer
from .colabfold import ColabFoldServer

# Deprecated servers (commented out for reference)
# from .reactome import ReactomeServer  # REMOVED: KEGG covers pathway + drug/compound
# from .pymol import PyMolServer  # REMOVED: Local visualization, not needed for automation
# from .scholar import ScholarServer  # REMOVED: NCBI PubMed covers literature search

__all__ = [
    # Original servers
    "KEGGServer",
    "RosettaServer",
    "NCBIServer",
    "ESMFoldServer",
    "UniProtServer",
    "BLASTServer",
    "PandasAnalysisServer",
    # New servers - v1
    "STRINGDBServer",
    "ChEMBLServer",
    "NanoporeServer",
    "MSAServer",
    "FoldseekServer",
    # New servers - v2 (Problem 3 & 4)
    "RCSBPDBServer",
    "InterProServer",
    "GProfilerServer",
    "OpenTargetsServer",
    "NetworkXServer",
    "IEDBServer",
    "VinaServer",
    "ProteinMPNNServer",
    "RFdiffusionServer",
    "ColabFoldServer",
]
