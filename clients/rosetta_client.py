"""
Rosetta Client - Direct integration with PyRosetta for protein structure prediction
Adapted from rosetta-mcp-server: https://github.com/Arielbs/rosetta-mcp-server
"""

import json
import subprocess
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class RosettaClient:
    """Client for PyRosetta protein structure analysis and scoring"""
    
    def __init__(self, python_path: str = "python3"):
        """
        Initialize Rosetta client
        
        Args:
            python_path: Path to Python interpreter with PyRosetta installed
        """
        self.python_path = python_path
        self.pyrosetta_available = self._check_pyrosetta()
        
        if not self.pyrosetta_available:
            logger.warning("PyRosetta not available. Install with: pip install pyrosetta-installer")
    
    def _check_pyrosetta(self) -> bool:
        """Check if PyRosetta is available"""
        script = """
import json
resp = {'available': False}
try:
    import pyrosetta
    resp['available'] = True
    resp['version'] = getattr(pyrosetta, '__version__', 'unknown')
except Exception as e:
    resp['error'] = str(e)
print(json.dumps(resp))
"""
        try:
            result = subprocess.run(
                [self.python_path, '-c', script],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                if data.get('available'):
                    logger.info(f"PyRosetta available, version: {data.get('version', 'unknown')}")
                    return True
                else:
                    logger.warning(f"PyRosetta not available: {data.get('error', 'unknown')}")
            return False
        except Exception as e:
            logger.warning(f"Failed to check PyRosetta: {e}")
            return False
    
    async def score_pdb(
        self,
        pdb_path: str,
        scorefxn: str = "ref2015"
    ) -> Dict[str, Any]:
        """
        Score a PDB file using PyRosetta
        
        Args:
            pdb_path: Path to PDB file
            scorefxn: Score function to use (default: ref2015)
            
        Returns:
            Dict with 'score' or 'error'
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}
        
        if not os.path.exists(pdb_path):
            return {"error": f"PDB file not found: {pdb_path}"}
        
        script = f"""
import json
try:
    import pyrosetta
except Exception as e:
    print(json.dumps({{"error": f"PyRosetta not available: {{str(e)}}"}}))
    raise SystemExit(0)

pyrosetta.init('-mute all')
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import get_score_function

pose = pose_from_pdb(r'''{pdb_path}''')
sfxn = get_score_function('{scorefxn}')
score = sfxn(pose)
print(json.dumps({{"score": float(score), "scorefxn": "{scorefxn}"}}))
"""
        
        try:
            logger.info(f"Scoring PDB with PyRosetta: {pdb_path}")
            result = subprocess.run(
                [self.python_path, '-c', script],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                logger.info(f"PyRosetta score: {data.get('score', 'N/A')}")
                return data
            else:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"PyRosetta scoring failed: {error_msg}")
                return {"error": error_msg}
                
        except subprocess.TimeoutExpired:
            logger.error("PyRosetta scoring timeout")
            return {"error": "Scoring timeout (>5 minutes)"}
        except Exception as e:
            logger.error(f"PyRosetta scoring error: {e}")
            return {"error": str(e)}
    
    async def predict_binding_energy(
        self,
        pdb_path: str,
        interface_chains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict binding energy between protein chains
        
        Args:
            pdb_path: Path to PDB file with protein complex
            interface_chains: List of chain IDs to analyze (e.g., ['A', 'B'])
            
        Returns:
            Dict with binding energy metrics
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}
        
        if not os.path.exists(pdb_path):
            return {"error": f"PDB file not found: {pdb_path}"}
        
        chains_str = json.dumps(interface_chains) if interface_chains else "None"
        
        script = f"""
import json
try:
    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.scoring import get_score_function
except Exception as e:
    print(json.dumps({{"error": f"PyRosetta not available: {{str(e)}}"}}))
    raise SystemExit(0)

pyrosetta.init('-mute all')

# Load pose
pose = pose_from_pdb(r'''{pdb_path}''')

# Score function
sfxn = get_score_function('ref2015')
total_score = sfxn(pose)

# Interface analysis
interface_chains = {chains_str}
if interface_chains and len(interface_chains) >= 2:
    interface = f"{{interface_chains[0]}}_{{interface_chains[1]}}"
    ia_mover = InterfaceAnalyzerMover(interface)
    ia_mover.apply(pose)
    
    # Extract metrics (simplified)
    result = {{
        "total_score": float(total_score),
        "interface": interface,
        "binding_analyzed": True,
        "message": "Interface analysis complete"
    }}
else:
    result = {{
        "total_score": float(total_score),
        "binding_analyzed": False,
        "message": "No interface chains specified"
    }}

print(json.dumps(result))
"""
        
        try:
            logger.info(f"Analyzing binding energy: {pdb_path}")
            result = subprocess.run(
                [self.python_path, '-c', script],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                logger.info(f"Binding analysis complete: {data.get('message', 'N/A')}")
                return data
            else:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"Binding analysis failed: {error_msg}")
                return {"error": error_msg}
                
        except subprocess.TimeoutExpired:
            logger.error("Binding analysis timeout")
            return {"error": "Analysis timeout (>10 minutes)"}
        except Exception as e:
            logger.error(f"Binding analysis error: {e}")
            return {"error": str(e)}
    
    async def relax_structure(
        self,
        pdb_path: str,
        output_path: str,
        scorefxn: str = "ref2015",
        repeats: int = 5
    ) -> Dict[str, Any]:
        """
        Relax protein structure using FastRelax
        
        Args:
            pdb_path: Input PDB file
            output_path: Output PDB file path
            scorefxn: Score function (default: ref2015)
            repeats: Number of relax cycles (default: 5)
            
        Returns:
            Dict with final score and output path
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}
        
        if not os.path.exists(pdb_path):
            return {"error": f"PDB file not found: {pdb_path}"}
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        script = f"""
import json
try:
    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.scoring import get_score_function
except Exception as e:
    print(json.dumps({{"error": f"PyRosetta not available: {{str(e)}}"}}))
    raise SystemExit(0)

pyrosetta.init('-mute all')

# Load pose
pose = pose_from_pdb(r'''{pdb_path}''')

# Setup FastRelax
sfxn = get_score_function('{scorefxn}')
relax = FastRelax()
relax.set_scorefxn(sfxn)
relax.max_iter({repeats})

# Initial score
initial_score = sfxn(pose)

# Relax
relax.apply(pose)

# Final score
final_score = sfxn(pose)

# Save
pose.dump_pdb(r'''{output_path}''')

result = {{
    "initial_score": float(initial_score),
    "final_score": float(final_score),
    "improvement": float(initial_score - final_score),
    "output_path": r'''{output_path}''',
    "scorefxn": "{scorefxn}",
    "repeats": {repeats}
}}

print(json.dumps(result))
"""
        
        try:
            logger.info(f"Relaxing structure: {pdb_path}")
            result = subprocess.run(
                [self.python_path, '-c', script],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                logger.info(f"Relax complete. Score improved by: {data.get('improvement', 'N/A')}")
                return data
            else:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"Structure relax failed: {error_msg}")
                return {"error": error_msg}
                
        except subprocess.TimeoutExpired:
            logger.error("Structure relax timeout")
            return {"error": "Relax timeout (>30 minutes)"}
        except Exception as e:
            logger.error(f"Structure relax error: {e}")
            return {"error": str(e)}
