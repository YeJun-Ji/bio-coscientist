"""
MSA (Multiple Sequence Alignment) MCP Server

Provides multiple sequence alignment and phylogenetic analysis using Clustal Omega web service.

Tools:
- align_sequences: Perform multiple sequence alignment
- build_phylogenetic_tree: Build phylogenetic tree from alignment
- calculate_conservation: Calculate sequence conservation scores
- get_consensus_sequence: Get consensus sequence from alignment
- get_pairwise_identity: Calculate pairwise sequence identity matrix
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from collections import Counter

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("msa-mcp")

# EMBL-EBI Clustal Omega API endpoints
CLUSTALO_RUN_URL = "https://www.ebi.ac.uk/Tools/services/rest/clustalo/run"
CLUSTALO_STATUS_URL = "https://www.ebi.ac.uk/Tools/services/rest/clustalo/status"
CLUSTALO_RESULT_URL = "https://www.ebi.ac.uk/Tools/services/rest/clustalo/result"

# Create MCP server instance
app = Server("msa")


def submit_clustalo_job(
    sequences: List[Dict[str, str]],
    email: str = "biocoscientist@example.com"
) -> Optional[str]:
    """Submit a Clustal Omega job to EMBL-EBI."""

    # Format sequences as FASTA
    fasta_str = ""
    for seq in sequences:
        seq_id = seq.get("id", f"seq_{len(fasta_str)}")
        sequence = seq.get("sequence", "")
        fasta_str += f">{seq_id}\n{sequence}\n"

    data = {
        "email": email,
        "sequence": fasta_str,
        "stype": "protein",  # or "dna"
        "outfmt": "clustal_num"  # clustal format with numbering
    }

    try:
        response = requests.post(CLUSTALO_RUN_URL, data=data, timeout=60)
        response.raise_for_status()
        job_id = response.text.strip()
        logger.info(f"Submitted Clustal Omega job: {job_id}")
        return job_id
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        return None


def check_job_status(job_id: str) -> str:
    """Check the status of a Clustal Omega job."""
    url = f"{CLUSTALO_STATUS_URL}/{job_id}"
    try:
        response = requests.get(url, timeout=30)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
        return "ERROR"


def get_job_result(job_id: str, result_type: str = "aln-clustal_num") -> Optional[str]:
    """Get the result of a Clustal Omega job."""
    url = f"{CLUSTALO_RESULT_URL}/{job_id}/{result_type}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Failed to get result: {e}")
        return None


def wait_for_job(job_id: str, max_wait: int = 300, poll_interval: int = 5) -> bool:
    """Wait for a job to complete."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = check_job_status(job_id)
        if status == "FINISHED":
            return True
        elif status in ["FAILURE", "ERROR", "NOT_FOUND"]:
            return False
        time.sleep(poll_interval)
    return False


def parse_clustal_alignment(alignment_text: str) -> Dict[str, str]:
    """Parse Clustal format alignment into dict."""
    sequences = {}
    for line in alignment_text.split("\n"):
        if line.startswith("CLUSTAL") or line.startswith(" ") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            seq_id = parts[0]
            seq_part = parts[1] if len(parts) > 1 else ""
            if seq_id not in sequences:
                sequences[seq_id] = ""
            sequences[seq_id] += seq_part.replace("-", "")
    return sequences


def parse_aligned_sequences(alignment_text: str) -> Dict[str, str]:
    """Parse Clustal format alignment keeping gaps."""
    sequences = {}
    for line in alignment_text.split("\n"):
        if line.startswith("CLUSTAL") or line.startswith(" ") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            seq_id = parts[0]
            seq_part = parts[1] if len(parts) > 1 else ""
            # Skip numbering at the end
            if seq_part.isdigit():
                continue
            if seq_id not in sequences:
                sequences[seq_id] = ""
            sequences[seq_id] += seq_part
    return sequences


def calculate_column_conservation(aligned_seqs: Dict[str, str]) -> List[float]:
    """Calculate conservation score for each column."""
    if not aligned_seqs:
        return []

    seq_list = list(aligned_seqs.values())
    if not seq_list:
        return []

    alignment_length = len(seq_list[0])
    conservation_scores = []

    for i in range(alignment_length):
        column = [seq[i] for seq in seq_list if i < len(seq)]
        # Remove gaps for conservation calculation
        column_no_gaps = [c for c in column if c != "-"]

        if not column_no_gaps:
            conservation_scores.append(0.0)
            continue

        # Calculate frequency of most common residue
        counter = Counter(column_no_gaps)
        most_common_count = counter.most_common(1)[0][1]
        conservation = most_common_count / len(column_no_gaps)
        conservation_scores.append(conservation)

    return conservation_scores


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MSA tools."""
    return [
        Tool(
            name="align_sequences",
            description="Perform multiple sequence alignment using Clustal Omega. USE FOR: Homolog comparison, conservation analysis, evolutionary study. ENTITY TYPES: protein, sequence. DATA FLOW: Requires sequence list, produces aligned sequences for conservation analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "sequence": {"type": "string"}
                            },
                            "required": ["id", "sequence"]
                        },
                        "description": "List of sequences with id and sequence"
                    },
                    "sequence_type": {
                        "type": "string",
                        "description": "Type of sequences",
                        "enum": ["protein", "dna"],
                        "default": "protein"
                    }
                },
                "required": ["sequences"]
            }
        ),
        Tool(
            name="calculate_conservation",
            description="Calculate per-residue conservation scores. USE FOR: Binding site identification, functional residue prediction, mutation impact assessment. ENTITY TYPES: protein, sequence. DATA FLOW: Produces conservation scores and highly conserved positions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "sequence": {"type": "string"}
                            },
                            "required": ["id", "sequence"]
                        },
                        "description": "List of sequences to align and analyze"
                    }
                },
                "required": ["sequences"]
            }
        ),
        Tool(
            name="get_consensus_sequence",
            description="Get consensus sequence from MSA. USE FOR: Representative sequence generation, family signature identification. ENTITY TYPES: protein, sequence. DATA FLOW: Produces consensus with identity scores per sequence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "sequence": {"type": "string"}
                            },
                            "required": ["id", "sequence"]
                        },
                        "description": "List of sequences to align"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum frequency for consensus (default: 0.5)",
                        "default": 0.5
                    }
                },
                "required": ["sequences"]
            }
        ),
        Tool(
            name="get_pairwise_identity",
            description="Calculate pairwise sequence identity matrix. USE FOR: Sequence diversity analysis, redundancy assessment, homolog clustering. ENTITY TYPES: protein, sequence. DATA FLOW: Produces identity matrix for evolutionary distance estimation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "sequence": {"type": "string"}
                            },
                            "required": ["id", "sequence"]
                        },
                        "description": "List of sequences to compare"
                    }
                },
                "required": ["sequences"]
            }
        ),
        Tool(
            name="build_distance_tree",
            description="Build UPGMA phylogenetic tree from sequences. USE FOR: Evolutionary relationship visualization, species comparison. ENTITY TYPES: protein, sequence. DATA FLOW: Produces Newick format tree for phylogenetic analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "sequence": {"type": "string"}
                            },
                            "required": ["id", "sequence"]
                        },
                        "description": "List of sequences for tree building"
                    }
                },
                "required": ["sequences"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "align_sequences":
            result = await align_sequences(
                sequences=arguments["sequences"],
                sequence_type=arguments.get("sequence_type", "protein")
            )
        elif name == "calculate_conservation":
            result = await calculate_conservation(
                sequences=arguments["sequences"]
            )
        elif name == "get_consensus_sequence":
            result = await get_consensus_sequence(
                sequences=arguments["sequences"],
                threshold=arguments.get("threshold", 0.5)
            )
        elif name == "get_pairwise_identity":
            result = await get_pairwise_identity(
                sequences=arguments["sequences"]
            )
        elif name == "build_distance_tree":
            result = await build_distance_tree(
                sequences=arguments["sequences"]
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def align_sequences(
    sequences: List[Dict[str, str]],
    sequence_type: str = "protein"
) -> Dict[str, Any]:
    """Perform multiple sequence alignment using Clustal Omega."""

    if len(sequences) < 2:
        return {"error": "Need at least 2 sequences for alignment"}

    # Submit job
    job_id = submit_clustalo_job(sequences)
    if not job_id:
        return {"error": "Failed to submit alignment job"}

    # Wait for completion
    logger.info(f"Waiting for alignment job {job_id}...")
    if not wait_for_job(job_id):
        return {"error": f"Alignment job {job_id} failed or timed out"}

    # Get result
    alignment = get_job_result(job_id, "aln-clustal_num")
    if not alignment:
        return {"error": "Failed to retrieve alignment"}

    # Parse alignment
    aligned_seqs = parse_aligned_sequences(alignment)

    return {
        "job_id": job_id,
        "sequence_count": len(sequences),
        "alignment_length": len(list(aligned_seqs.values())[0]) if aligned_seqs else 0,
        "aligned_sequences": aligned_seqs,
        "clustal_format": alignment[:2000] + "..." if len(alignment) > 2000 else alignment
    }


async def calculate_conservation(
    sequences: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Calculate conservation scores for aligned sequences."""

    # First align
    align_result = await align_sequences(sequences)
    if "error" in align_result:
        return align_result

    aligned_seqs = align_result.get("aligned_sequences", {})
    if not aligned_seqs:
        return {"error": "No aligned sequences available"}

    # Calculate conservation
    scores = calculate_column_conservation(aligned_seqs)

    # Identify highly conserved positions
    conserved_positions = [
        {"position": i + 1, "score": score}
        for i, score in enumerate(scores)
        if score >= 0.9
    ]

    # Calculate overall conservation
    avg_conservation = sum(scores) / len(scores) if scores else 0

    return {
        "sequence_count": len(aligned_seqs),
        "alignment_length": len(scores),
        "average_conservation": avg_conservation,
        "highly_conserved_count": len(conserved_positions),
        "highly_conserved_positions": conserved_positions[:50],  # Top 50
        "conservation_distribution": {
            "100%": sum(1 for s in scores if s == 1.0),
            ">=90%": sum(1 for s in scores if s >= 0.9),
            ">=70%": sum(1 for s in scores if s >= 0.7),
            ">=50%": sum(1 for s in scores if s >= 0.5),
            "<50%": sum(1 for s in scores if s < 0.5)
        }
    }


async def get_consensus_sequence(
    sequences: List[Dict[str, str]],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Get consensus sequence from alignment."""

    # First align
    align_result = await align_sequences(sequences)
    if "error" in align_result:
        return align_result

    aligned_seqs = align_result.get("aligned_sequences", {})
    if not aligned_seqs:
        return {"error": "No aligned sequences available"}

    seq_list = list(aligned_seqs.values())
    alignment_length = len(seq_list[0]) if seq_list else 0

    consensus = ""
    for i in range(alignment_length):
        column = [seq[i] for seq in seq_list if i < len(seq)]
        column_no_gaps = [c for c in column if c != "-"]

        if not column_no_gaps:
            consensus += "-"
            continue

        counter = Counter(column_no_gaps)
        most_common, count = counter.most_common(1)[0]
        freq = count / len(column_no_gaps)

        if freq >= threshold:
            consensus += most_common
        else:
            consensus += "X"  # Ambiguous

    return {
        "consensus_sequence": consensus,
        "length": len(consensus),
        "threshold": threshold,
        "sequence_count": len(aligned_seqs),
        "identity_to_consensus": {
            seq_id: sum(1 for a, b in zip(seq, consensus) if a == b and a != "-" and b != "X") / len(consensus.replace("-", "").replace("X", ""))
            for seq_id, seq in aligned_seqs.items()
        } if consensus.replace("-", "").replace("X", "") else {}
    }


async def get_pairwise_identity(
    sequences: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Calculate pairwise sequence identity matrix."""

    # First align
    align_result = await align_sequences(sequences)
    if "error" in align_result:
        return align_result

    aligned_seqs = align_result.get("aligned_sequences", {})
    if not aligned_seqs:
        return {"error": "No aligned sequences available"}

    seq_ids = list(aligned_seqs.keys())
    identity_matrix = {}

    for i, id1 in enumerate(seq_ids):
        identity_matrix[id1] = {}
        seq1 = aligned_seqs[id1]

        for id2 in seq_ids:
            seq2 = aligned_seqs[id2]

            # Calculate identity (excluding gap-gap positions)
            matches = 0
            compared = 0

            for a, b in zip(seq1, seq2):
                if a == "-" and b == "-":
                    continue
                compared += 1
                if a == b:
                    matches += 1

            identity = matches / compared if compared > 0 else 0
            identity_matrix[id1][id2] = round(identity, 4)

    return {
        "sequence_count": len(seq_ids),
        "identity_matrix": identity_matrix,
        "average_identity": sum(
            identity_matrix[id1][id2]
            for id1 in seq_ids
            for id2 in seq_ids
            if id1 != id2
        ) / (len(seq_ids) * (len(seq_ids) - 1)) if len(seq_ids) > 1 else 1.0
    }


async def build_distance_tree(
    sequences: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Build a simple UPGMA distance tree."""

    # Get pairwise identity
    identity_result = await get_pairwise_identity(sequences)
    if "error" in identity_result:
        return identity_result

    identity_matrix = identity_result.get("identity_matrix", {})
    seq_ids = list(identity_matrix.keys())

    # Convert identity to distance (1 - identity)
    distance_matrix = {
        id1: {id2: 1 - identity_matrix[id1][id2] for id2 in seq_ids}
        for id1 in seq_ids
    }

    # Simple UPGMA clustering
    clusters = [[seq_id] for seq_id in seq_ids]
    newick_parts = {seq_id: seq_id for seq_id in seq_ids}

    while len(clusters) > 1:
        # Find minimum distance pair
        min_dist = float('inf')
        min_i, min_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Average distance between clusters
                dist_sum = 0
                count = 0
                for id1 in clusters[i]:
                    for id2 in clusters[j]:
                        dist_sum += distance_matrix[id1][id2]
                        count += 1
                avg_dist = dist_sum / count if count > 0 else float('inf')

                if avg_dist < min_dist:
                    min_dist = avg_dist
                    min_i, min_j = i, j

        # Merge clusters
        new_cluster = clusters[min_i] + clusters[min_j]

        # Create newick for merged cluster
        newick_i = newick_parts[clusters[min_i][0]]
        newick_j = newick_parts[clusters[min_j][0]]
        new_newick = f"({newick_i}:{min_dist/2:.4f},{newick_j}:{min_dist/2:.4f})"

        # Update data structures
        for member in new_cluster:
            newick_parts[member] = new_newick

        # Remove merged clusters and add new one
        clusters = [c for i, c in enumerate(clusters) if i not in [min_i, min_j]]
        clusters.append(new_cluster)

    # Final newick tree
    final_newick = newick_parts[clusters[0][0]] + ";"

    return {
        "tree_format": "newick",
        "tree": final_newick,
        "sequence_count": len(seq_ids),
        "method": "UPGMA",
        "distance_matrix": distance_matrix,
        "note": "Simple UPGMA tree. For publication-quality trees, use dedicated phylogenetics software."
    }


async def main():
    """Run the MCP server."""
    logger.info("Starting MSA MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
