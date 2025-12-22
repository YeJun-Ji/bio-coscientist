"""
ESMFold MCP Server

Provides protein structure prediction using the ESMFold API.
Returns pLDDT confidence scores for structure quality assessment.

Tools:
- fold_sequence: Predict 3D structure from protein sequence
- get_plddt_score: Get pLDDT confidence score
- validate_structure: Comprehensive structure quality validation
"""

import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("esmfold-mcp")

# ESMFold API endpoint
ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_SEQUENCE_LENGTH = 400

# Create MCP server instance
app = Server("esmfold")


def validate_sequence(sequence: str) -> tuple[bool, str]:
    """
    Validate protein sequence.

    Returns:
        (is_valid, error_message)
    """
    # Remove whitespace
    sequence = sequence.strip().upper()

    # Check length
    if len(sequence) == 0:
        return False, "Empty sequence provided"

    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return False, f"Sequence too long: {len(sequence)} residues (max {MAX_SEQUENCE_LENGTH})"

    # Check for valid amino acid characters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_chars = set(sequence) - valid_aa

    if invalid_chars:
        return False, f"Invalid amino acid characters: {invalid_chars}"

    return True, ""


def parse_plddt_from_pdb(pdb_content: str) -> Dict[str, Any]:
    """
    Extract pLDDT scores from PDB file B-factor column.

    In ESMFold PDB output, pLDDT scores are stored in the B-factor column
    for each CA (alpha carbon) atom.

    Note: ESMAtlas API returns pLDDT on 0-1 scale, so we multiply by 100 if needed.

    Returns:
        {
            "per_residue": [85.2, 90.1, ...],
            "mean_plddt": 87.5,
            "min_plddt": 65.3,
            "max_plddt": 95.8,
            "low_confidence_regions": [(start, end), ...]
        }
    """
    plddt_scores = []
    residue_positions = []

    for line in pdb_content.split("\n"):
        # ATOM records for CA atoms contain pLDDT in B-factor column (columns 61-66)
        if line.startswith("ATOM") and " CA " in line:
            try:
                # B-factor is in columns 61-66 (0-indexed: 60:66)
                bfactor = float(line[60:66].strip())
                residue_num = int(line[22:26].strip())
                plddt_scores.append(bfactor)
                residue_positions.append(residue_num)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse B-factor from line: {line[:30]}... Error: {e}")
                continue

    if not plddt_scores:
        return {
            "per_residue": [],
            "mean_plddt": 0.0,
            "min_plddt": 0.0,
            "max_plddt": 0.0,
            "low_confidence_regions": [],
            "error": "No pLDDT scores found in PDB"
        }

    # ESMAtlas API returns pLDDT on 0-1 scale, convert to 0-100 if needed
    if max(plddt_scores) <= 1.0:
        plddt_scores = [score * 100 for score in plddt_scores]

    # Calculate statistics
    mean_plddt = sum(plddt_scores) / len(plddt_scores)
    min_plddt = min(plddt_scores)
    max_plddt = max(plddt_scores)

    # Find low confidence regions (pLDDT < 50)
    low_confidence_regions = []
    in_low_region = False
    region_start = None

    for i, score in enumerate(plddt_scores):
        if score < 50:
            if not in_low_region:
                region_start = residue_positions[i]
                in_low_region = True
        else:
            if in_low_region:
                low_confidence_regions.append((region_start, residue_positions[i-1]))
                in_low_region = False

    if in_low_region:
        low_confidence_regions.append((region_start, residue_positions[-1]))

    return {
        "per_residue": plddt_scores,
        "mean_plddt": round(mean_plddt, 2),
        "min_plddt": round(min_plddt, 2),
        "max_plddt": round(max_plddt, 2),
        "low_confidence_regions": low_confidence_regions,
        "num_residues": len(plddt_scores)
    }


def call_esmfold_api(sequence: str) -> tuple[Optional[str], Optional[str]]:
    """
    Call ESMFold API to predict structure.

    Returns:
        (pdb_content, error_message)
    """
    try:
        logger.info(f"Calling ESMFold API for sequence of length {len(sequence)}")

        response = requests.post(
            ESMFOLD_API,
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=120  # 2 minute timeout
        )

        if response.status_code == 200:
            logger.info("ESMFold API call successful")
            return response.text, None
        else:
            error_msg = f"ESMFold API error: {response.status_code} - {response.text[:200]}"
            logger.error(error_msg)
            return None, error_msg

    except requests.exceptions.Timeout:
        return None, "ESMFold API timeout (>120s)"
    except requests.exceptions.RequestException as e:
        return None, f"ESMFold API request failed: {str(e)}"


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="fold_sequence",
            description="Predict 3D structure from sequence using ESMFold (fast, no GPU). USE FOR: Quick structure prediction, design validation, single protein modeling. ENTITY TYPES: protein, sequence, structure. DATA FLOW: Produces PDB with pLDDT scores. Max 400 residues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein sequence in single-letter amino acid code (e.g., 'MVHLTPEEKSAVTALWGKV...'). Maximum 400 residues."
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="get_plddt_score",
            description="Get pLDDT confidence score for structure quality. USE FOR: Fold quality assessment, design ranking, structure reliability check. ENTITY TYPES: protein, sequence. DATA FLOW: Produces per-residue pLDDT (>70 reliable, >90 excellent).",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein sequence in single-letter amino acid code. Maximum 400 residues."
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="validate_structure",
            description="Comprehensive structure quality validation with grade. USE FOR: Design quality assessment, fold verification, structure validation. ENTITY TYPES: protein, sequence, structure. DATA FLOW: Produces quality grade (excellent/good/moderate/poor) and warnings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein sequence in single-letter amino acid code. Maximum 400 residues."
                    }
                },
                "required": ["sequence"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "fold_sequence":
        return await fold_sequence(arguments.get("sequence", ""))
    elif name == "get_plddt_score":
        return await get_plddt_score(arguments.get("sequence", ""))
    elif name == "validate_structure":
        return await validate_structure(arguments.get("sequence", ""))
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def fold_sequence(sequence: str) -> List[TextContent]:
    """Predict 3D structure from protein sequence."""

    # Clean sequence
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")

    # Validate
    is_valid, error = validate_sequence(sequence)
    if not is_valid:
        return [TextContent(type="text", text=json.dumps({"error": error}))]

    # Call ESMFold API
    pdb_content, error = call_esmfold_api(sequence)

    if error:
        return [TextContent(type="text", text=json.dumps({"error": error}))]

    return [TextContent(type="text", text=pdb_content)]


async def get_plddt_score(sequence: str) -> List[TextContent]:
    """Get pLDDT confidence scores for a protein sequence."""

    # Clean sequence
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")

    # Validate
    is_valid, error = validate_sequence(sequence)
    if not is_valid:
        return [TextContent(type="text", text=json.dumps({"error": error}))]

    # Call ESMFold API
    pdb_content, error = call_esmfold_api(sequence)

    if error:
        return [TextContent(type="text", text=json.dumps({"error": error}))]

    # Parse pLDDT from PDB
    plddt_data = parse_plddt_from_pdb(pdb_content)

    return [TextContent(type="text", text=json.dumps(plddt_data))]


async def validate_structure(sequence: str) -> List[TextContent]:
    """
    Comprehensive structure quality validation.

    Returns quality grade, pLDDT scores, and warnings.
    """

    # Clean sequence
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")

    # Validate
    is_valid, error = validate_sequence(sequence)
    if not is_valid:
        return [TextContent(type="text", text=json.dumps({
            "valid": False,
            "error": error,
            "quality": "invalid"
        }))]

    # Call ESMFold API
    pdb_content, error = call_esmfold_api(sequence)

    if error:
        return [TextContent(type="text", text=json.dumps({
            "valid": False,
            "error": error,
            "quality": "api_error"
        }))]

    # Parse pLDDT
    plddt_data = parse_plddt_from_pdb(pdb_content)
    mean_plddt = plddt_data.get("mean_plddt", 0)

    # Determine quality grade
    if mean_plddt >= 90:
        quality = "excellent"
        pass_threshold = True
    elif mean_plddt >= 70:
        quality = "good"
        pass_threshold = True
    elif mean_plddt >= 50:
        quality = "moderate"
        pass_threshold = False
    else:
        quality = "poor"
        pass_threshold = False

    # Generate warnings
    warnings = []

    if plddt_data.get("low_confidence_regions"):
        regions = plddt_data["low_confidence_regions"]
        warnings.append(f"Low confidence regions (pLDDT<50): {regions}")

    if mean_plddt < 70:
        warnings.append("Mean pLDDT below 70 - structure may not be reliable")

    if len(sequence) > 300:
        warnings.append("Long sequence - structure prediction accuracy may decrease")

    result = {
        "valid": True,
        "sequence_length": len(sequence),
        "mean_plddt": mean_plddt,
        "min_plddt": plddt_data.get("min_plddt", 0),
        "max_plddt": plddt_data.get("max_plddt", 0),
        "quality": quality,
        "pass_threshold": pass_threshold,
        "low_confidence_regions": plddt_data.get("low_confidence_regions", []),
        "warnings": warnings,
        "interpretation": {
            "excellent": "pLDDT >= 90: Very high confidence, suitable for detailed analysis",
            "good": "pLDDT 70-90: Confident prediction, reliable for most purposes",
            "moderate": "pLDDT 50-70: Low confidence, use with caution",
            "poor": "pLDDT < 50: Very low confidence, likely disordered region"
        }[quality]
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run the MCP server."""
    logger.info("Starting ESMFold MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
