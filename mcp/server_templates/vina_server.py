"""
AutoDock Vina MCP Server

Provides molecular docking and binding affinity prediction.
Uses the Python vina package for docking.

Tools:
- dock_ligand: Dock ligand to receptor
- prepare_receptor: Prepare receptor for docking
- prepare_ligand: Prepare ligand from SMILES
- calculate_binding_affinity: Get binding scores
"""

import asyncio
import json
import logging
import tempfile
import os
from typing import List, Dict, Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vina-mcp")

# Create MCP server instance
app = Server("vina")

# Check for optional dependencies
VINA_AVAILABLE = False
RDKIT_AVAILABLE = False
MEEKO_AVAILABLE = False

try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    logger.warning("vina package not available")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("rdkit package not available")

try:
    import meeko
    MEEKO_AVAILABLE = True
except ImportError:
    logger.warning("meeko package not available")


def smiles_to_pdbqt(smiles: str) -> tuple[Optional[str], Optional[str]]:
    """
    Convert SMILES to PDBQT format for docking.

    Returns:
        (pdbqt_content, error_message)
    """
    if not RDKIT_AVAILABLE:
        return None, "RDKit not available. Install with: pip install rdkit"

    if not MEEKO_AVAILABLE:
        return None, "Meeko not available. Install with: pip install meeko"

    try:
        # Generate 3D structure from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"Invalid SMILES: {smiles}"

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            return None, "Failed to generate 3D coordinates"

        # Optimize geometry
        AllChem.MMFFOptimizeMolecule(mol)

        # Convert to PDBQT using meeko
        from meeko import MoleculePreparation
        preparator = MoleculePreparation()
        preparator.prepare(mol)

        pdbqt_string = preparator.write_pdbqt_string()

        return pdbqt_string, None

    except Exception as e:
        return None, f"Conversion failed: {str(e)}"


def pdb_to_pdbqt(pdb_content: str) -> tuple[Optional[str], Optional[str]]:
    """
    Convert PDB to PDBQT format (simplified conversion).

    Note: This is a simplified conversion. For production use,
    consider using AutoDockTools or OpenBabel.

    Returns:
        (pdbqt_content, error_message)
    """
    try:
        pdbqt_lines = []

        for line in pdb_content.split("\n"):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Add partial charge column (placeholder)
                # Real PDBQT needs proper charges from tools like AutoDockTools
                atom_type = line[76:78].strip() if len(line) > 76 else line[12:14].strip()

                # Simple atom type mapping
                if atom_type in ["C", "CA", "CB", "CG"]:
                    ad_type = "C"
                elif atom_type in ["N", "NA", "NE"]:
                    ad_type = "N"
                elif atom_type in ["O", "OE", "OG"]:
                    ad_type = "OA"
                elif atom_type in ["S", "SG"]:
                    ad_type = "SA"
                elif atom_type == "H":
                    ad_type = "HD"
                else:
                    ad_type = atom_type[:2] if len(atom_type) > 1 else atom_type + " "

                # Format PDBQT line (add charge=0.0 and atom type)
                pdbqt_line = f"{line[:54]}  0.00 {ad_type:>2}"
                pdbqt_lines.append(pdbqt_line)

            elif line.startswith("END") or line.startswith("TER"):
                pdbqt_lines.append(line)

        return "\n".join(pdbqt_lines), None

    except Exception as e:
        return None, f"Conversion failed: {str(e)}"


def run_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple,
    size: tuple,
    exhaustiveness: int = 8,
    n_poses: int = 5
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Run Vina docking.

    Args:
        receptor_pdbqt: Receptor in PDBQT format
        ligand_pdbqt: Ligand in PDBQT format
        center: (x, y, z) center of search box
        size: (x, y, z) size of search box
        exhaustiveness: Search exhaustiveness (default: 8)
        n_poses: Number of poses to return (default: 5)

    Returns:
        (docking_results, error_message)
    """
    if not VINA_AVAILABLE:
        return None, "Vina not available. Install with: pip install vina"

    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
            f.write(receptor_pdbqt)
            receptor_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
            f.write(ligand_pdbqt)
            ligand_path = f.name

        output_path = tempfile.mktemp(suffix='.pdbqt')

        try:
            # Initialize Vina
            v = Vina(sf_name='vina')

            # Set receptor
            v.set_receptor(receptor_path)

            # Set ligand
            v.set_ligand_from_file(ligand_path)

            # Compute maps
            v.compute_vina_maps(center=list(center), box_size=list(size))

            # Run docking
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

            # Get energies
            energies = v.energies()

            # Write output
            v.write_poses(output_path, n_poses=n_poses, overwrite=True)

            # Read output poses
            with open(output_path, 'r') as f:
                poses_pdbqt = f.read()

            # Parse results
            poses = []
            for i, energy in enumerate(energies):
                poses.append({
                    "pose": i + 1,
                    "affinity_kcal_mol": round(energy[0], 2),
                    "rmsd_lb": round(energy[1], 2) if len(energy) > 1 else None,
                    "rmsd_ub": round(energy[2], 2) if len(energy) > 2 else None
                })

            return {
                "num_poses": len(poses),
                "best_affinity": poses[0]["affinity_kcal_mol"] if poses else None,
                "poses": poses,
                "docked_poses_pdbqt": poses_pdbqt,
                "search_center": center,
                "search_size": size,
                "interpretation": {
                    "< -10 kcal/mol": "Excellent binding",
                    "-10 to -7 kcal/mol": "Good binding",
                    "-7 to -5 kcal/mol": "Moderate binding",
                    "> -5 kcal/mol": "Weak binding"
                }
            }, None

        finally:
            # Cleanup
            for path in [receptor_path, ligand_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    except Exception as e:
        return None, f"Docking failed: {str(e)}"


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="dock_ligand",
            description="Dock small molecule to protein receptor. USE FOR: Drug binding prediction, virtual screening, lead optimization. ENTITY TYPES: compound, protein, structure. DATA FLOW: Requires receptor PDB and ligand SMILES, produces binding poses and affinity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "receptor_pdb": {
                        "type": "string",
                        "description": "Receptor structure in PDB format"
                    },
                    "ligand_smiles": {
                        "type": "string",
                        "description": "Ligand SMILES string (e.g., 'CC(=O)Oc1ccccc1C(=O)O' for aspirin)"
                    },
                    "center_x": {"type": "number", "description": "X coordinate of search box center"},
                    "center_y": {"type": "number", "description": "Y coordinate of search box center"},
                    "center_z": {"type": "number", "description": "Z coordinate of search box center"},
                    "size_x": {"type": "number", "description": "Search box size in X (default: 20)"},
                    "size_y": {"type": "number", "description": "Search box size in Y (default: 20)"},
                    "size_z": {"type": "number", "description": "Search box size in Z (default: 20)"},
                    "exhaustiveness": {"type": "integer", "description": "Search exhaustiveness (default: 8)"}
                },
                "required": ["receptor_pdb", "ligand_smiles", "center_x", "center_y", "center_z"]
            }
        ),
        Tool(
            name="prepare_receptor",
            description="Convert receptor PDB to PDBQT for docking. USE FOR: Docking preparation, receptor setup. ENTITY TYPES: protein, structure. DATA FLOW: Requires PDB content, produces PDBQT for dock_ligand input.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_content": {
                        "type": "string",
                        "description": "Receptor structure in PDB format"
                    }
                },
                "required": ["pdb_content"]
            }
        ),
        Tool(
            name="prepare_ligand",
            description="Convert SMILES to PDBQT 3D structure for docking. USE FOR: Ligand preparation, compound setup. ENTITY TYPES: compound. DATA FLOW: Requires SMILES, produces 3D PDBQT for dock_ligand input.",
            inputSchema={
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Ligand SMILES string"
                    }
                },
                "required": ["smiles"]
            }
        ),
        Tool(
            name="calculate_binding_affinity",
            description="Convert docking score to estimated KD. USE FOR: Affinity estimation, hit prioritization. ENTITY TYPES: compound. DATA FLOW: Produces KD estimate from docking kcal/mol for compound ranking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "affinity_kcal_mol": {
                        "type": "number",
                        "description": "Binding affinity in kcal/mol (from docking)"
                    }
                },
                "required": ["affinity_kcal_mol"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "dock_ligand":
        receptor_pdb = arguments.get("receptor_pdb", "")
        ligand_smiles = arguments.get("ligand_smiles", "")
        center = (
            arguments.get("center_x", 0),
            arguments.get("center_y", 0),
            arguments.get("center_z", 0)
        )
        size = (
            arguments.get("size_x", 20),
            arguments.get("size_y", 20),
            arguments.get("size_z", 20)
        )
        exhaustiveness = arguments.get("exhaustiveness", 8)

        # Prepare receptor
        receptor_pdbqt, error = pdb_to_pdbqt(receptor_pdb)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": f"Receptor preparation: {error}"}))]

        # Prepare ligand
        ligand_pdbqt, error = smiles_to_pdbqt(ligand_smiles)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": f"Ligand preparation: {error}"}))]

        # Run docking
        result, error = run_docking(receptor_pdbqt, ligand_pdbqt, center, size, exhaustiveness)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "prepare_receptor":
        pdb_content = arguments.get("pdb_content", "")

        pdbqt, error = pdb_to_pdbqt(pdb_content)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=pdbqt)]

    elif name == "prepare_ligand":
        smiles = arguments.get("smiles", "")

        pdbqt, error = smiles_to_pdbqt(smiles)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=pdbqt)]

    elif name == "calculate_binding_affinity":
        affinity = arguments.get("affinity_kcal_mol", 0)

        # Convert kcal/mol to KD using: ΔG = RT ln(KD)
        # At 298K (25°C): KD = exp(ΔG / (RT))
        import math
        R = 1.987e-3  # kcal/(mol·K)
        T = 298.15  # K

        try:
            kd = math.exp(affinity / (R * T))

            # Determine units
            if kd < 1e-9:
                kd_formatted = f"{kd * 1e12:.2f} pM"
            elif kd < 1e-6:
                kd_formatted = f"{kd * 1e9:.2f} nM"
            elif kd < 1e-3:
                kd_formatted = f"{kd * 1e6:.2f} μM"
            else:
                kd_formatted = f"{kd * 1e3:.2f} mM"

            return [TextContent(type="text", text=json.dumps({
                "affinity_kcal_mol": affinity,
                "estimated_KD": kd_formatted,
                "KD_molar": kd,
                "temperature": "298.15 K (25°C)",
                "note": "This is an estimate. Actual KD may differ due to entropy, solvation, etc."
            }, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    logger.info("Starting Vina MCP Server...")

    # Log availability
    logger.info(f"Vina available: {VINA_AVAILABLE}")
    logger.info(f"RDKit available: {RDKIT_AVAILABLE}")
    logger.info(f"Meeko available: {MEEKO_AVAILABLE}")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
