"""
RCSB PDB MCP Server

Provides protein structure download and analysis using the RCSB PDB API.
Supports structure search, binding site analysis, and ligand interactions.

Tools:
- download_pdb: Download structure by PDB ID
- search_structures: Search structures by name/keyword
- get_binding_sites: Analyze binding sites in a structure
- get_ligand_interactions: Get ligand-protein interactions
- search_by_sequence: Search by sequence similarity
- get_protein_info: Get structure metadata
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rcsbpdb-mcp")

# RCSB PDB API endpoints
RCSB_DATA_API = "https://data.rcsb.org/rest/v1"
RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL_API = "https://data.rcsb.org/graphql"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download"

# Create MCP server instance
app = Server("rcsbpdb")


def download_structure(pdb_id: str, format: str = "pdb") -> tuple[Optional[str], Optional[str]]:
    """
    Download structure from RCSB PDB.

    Args:
        pdb_id: 4-letter PDB ID
        format: 'pdb' or 'cif'

    Returns:
        (content, error_message)
    """
    pdb_id = pdb_id.upper().strip()

    if len(pdb_id) != 4:
        return None, f"Invalid PDB ID format: {pdb_id} (must be 4 characters)"

    ext = "pdb" if format.lower() == "pdb" else "cif"
    url = f"{PDB_DOWNLOAD_URL}/{pdb_id}.{ext}"

    try:
        logger.info(f"Downloading {pdb_id}.{ext}")
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            return response.text, None
        elif response.status_code == 404:
            return None, f"PDB ID not found: {pdb_id}"
        else:
            return None, f"Download failed: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def search_by_text(query: str, max_results: int = 10) -> tuple[Optional[List[Dict]], Optional[str]]:
    """
    Search structures by text query.

    Returns:
        (results, error_message)
    """
    search_query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": query
            }
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }

    try:
        response = requests.post(
            RCSB_SEARCH_API,
            json=search_query,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            results = []
            for hit in data.get("result_set", []):
                results.append({
                    "pdb_id": hit.get("identifier"),
                    "score": hit.get("score")
                })
            return results, None
        else:
            return None, f"Search failed: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def search_by_sequence(sequence: str, identity_cutoff: float = 0.9, max_results: int = 10) -> tuple[Optional[List[Dict]], Optional[str]]:
    """
    Search structures by sequence similarity.

    Args:
        sequence: Protein sequence
        identity_cutoff: Minimum sequence identity (0-1)
        max_results: Maximum number of results

    Returns:
        (results, error_message)
    """
    search_query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": sequence.strip().upper()
            }
        },
        "return_type": "polymer_entity",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }

    try:
        response = requests.post(
            RCSB_SEARCH_API,
            json=search_query,
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            results = []
            for hit in data.get("result_set", []):
                identifier = hit.get("identifier", "")
                pdb_id = identifier.split("_")[0] if "_" in identifier else identifier
                results.append({
                    "pdb_id": pdb_id,
                    "entity_id": identifier,
                    "score": hit.get("score"),
                    "identity": hit.get("services", [{}])[0].get("nodes", [{}])[0].get("match_context", [{}])[0].get("sequence_identity") if hit.get("services") else None
                })
            return results, None
        else:
            return None, f"Search failed: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def get_entry_info(pdb_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get structure metadata using GraphQL.

    Returns:
        (info_dict, error_message)
    """
    pdb_id = pdb_id.upper().strip()

    query = """
    query ($id: String!) {
        entry(entry_id: $id) {
            rcsb_id
            struct {
                title
            }
            rcsb_entry_info {
                resolution_combined
                molecular_weight
                deposited_polymer_entity_instance_count
                polymer_entity_count
                nonpolymer_entity_count
            }
            exptl {
                method
            }
            rcsb_primary_citation {
                title
                journal_abbrev
                year
                pdbx_database_id_DOI
            }
            polymer_entities {
                rcsb_polymer_entity {
                    pdbx_description
                }
                entity_poly {
                    pdbx_seq_one_letter_code_can
                    type
                }
                rcsb_entity_source_organism {
                    scientific_name
                }
            }
        }
    }
    """

    try:
        response = requests.post(
            RCSB_GRAPHQL_API,
            json={"query": query, "variables": {"id": pdb_id}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            entry = data.get("data", {}).get("entry")

            if not entry:
                return None, f"PDB ID not found: {pdb_id}"

            # Extract info
            info = {
                "pdb_id": entry.get("rcsb_id"),
                "title": entry.get("struct", {}).get("title"),
                "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined"),
                "molecular_weight": entry.get("rcsb_entry_info", {}).get("molecular_weight"),
                "method": entry.get("exptl", [{}])[0].get("method") if entry.get("exptl") else None,
                "polymer_entities": []
            }

            # Add citation
            citation = entry.get("rcsb_primary_citation", {})
            if citation:
                info["citation"] = {
                    "title": citation.get("title"),
                    "journal": citation.get("journal_abbrev"),
                    "year": citation.get("year"),
                    "doi": citation.get("pdbx_database_id_DOI")
                }

            # Add polymer entities
            for entity in entry.get("polymer_entities", []):
                entity_info = {
                    "description": entity.get("rcsb_polymer_entity", {}).get("pdbx_description"),
                    "sequence": entity.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can"),
                    "type": entity.get("entity_poly", {}).get("type"),
                    "organism": entity.get("rcsb_entity_source_organism", [{}])[0].get("scientific_name") if entity.get("rcsb_entity_source_organism") else None
                }
                info["polymer_entities"].append(entity_info)

            return info, None
        else:
            return None, f"Query failed: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def get_binding_sites(pdb_id: str) -> tuple[Optional[List[Dict]], Optional[str]]:
    """
    Get binding site information for a structure.

    Returns:
        (binding_sites, error_message)
    """
    pdb_id = pdb_id.upper().strip()

    query = """
    query ($id: String!) {
        entry(entry_id: $id) {
            rcsb_binding_affinity {
                comp_id
                type
                value
                unit
                provenance_code
            }
            nonpolymer_entities {
                nonpolymer_comp {
                    chem_comp {
                        id
                        name
                        formula
                        formula_weight
                        type
                    }
                }
                rcsb_nonpolymer_entity_container_identifiers {
                    auth_asym_ids
                }
            }
            polymer_entities {
                rcsb_polymer_entity {
                    pdbx_description
                }
                rcsb_polymer_entity_container_identifiers {
                    auth_asym_ids
                }
            }
        }
    }
    """

    try:
        response = requests.post(
            RCSB_GRAPHQL_API,
            json={"query": query, "variables": {"id": pdb_id}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            entry = data.get("data", {}).get("entry")

            if not entry:
                return None, f"PDB ID not found: {pdb_id}"

            binding_sites = []

            # Get ligands (nonpolymer entities)
            for entity in entry.get("nonpolymer_entities", []):
                comp = entity.get("nonpolymer_comp", {}).get("chem_comp", {})
                chains = entity.get("rcsb_nonpolymer_entity_container_identifiers", {}).get("auth_asym_ids", [])

                # Skip water and common ions
                comp_id = comp.get("id", "")
                if comp_id in ["HOH", "WAT", "NA", "CL", "MG", "CA", "ZN", "K"]:
                    continue

                site = {
                    "ligand_id": comp_id,
                    "ligand_name": comp.get("name"),
                    "formula": comp.get("formula"),
                    "weight": comp.get("formula_weight"),
                    "type": comp.get("type"),
                    "chains": chains
                }
                binding_sites.append(site)

            # Add binding affinity data if available
            affinities = entry.get("rcsb_binding_affinity", []) or []
            for aff in affinities:
                comp_id = aff.get("comp_id")
                for site in binding_sites:
                    if site["ligand_id"] == comp_id:
                        site["affinity"] = {
                            "type": aff.get("type"),
                            "value": aff.get("value"),
                            "unit": aff.get("unit")
                        }

            # Add protein chain info
            protein_chains = []
            for entity in entry.get("polymer_entities", []):
                desc = entity.get("rcsb_polymer_entity", {}).get("pdbx_description")
                chains = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
                protein_chains.append({
                    "description": desc,
                    "chains": chains
                })

            return {
                "pdb_id": pdb_id,
                "ligands": binding_sites,
                "protein_chains": protein_chains,
                "num_ligands": len(binding_sites)
            }, None
        else:
            return None, f"Query failed: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def get_ligand_interactions(pdb_id: str, ligand_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get detailed ligand-protein interactions.

    This uses RCSB's ligand interaction data.

    Returns:
        (interactions, error_message)
    """
    pdb_id = pdb_id.upper().strip()
    ligand_id = ligand_id.upper().strip()

    # Use the data API for ligand info
    url = f"{RCSB_DATA_API}/core/chemcomp/{ligand_id}"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            ligand_data = response.json()

            result = {
                "pdb_id": pdb_id,
                "ligand_id": ligand_id,
                "ligand_info": {
                    "name": ligand_data.get("chem_comp", {}).get("name"),
                    "formula": ligand_data.get("chem_comp", {}).get("formula"),
                    "weight": ligand_data.get("chem_comp", {}).get("formula_weight"),
                    "type": ligand_data.get("chem_comp", {}).get("type"),
                    "smiles": ligand_data.get("rcsb_chem_comp_descriptor", {}).get("smiles") if ligand_data.get("rcsb_chem_comp_descriptor") else None
                },
                "note": "For detailed atom-level interactions, analyze the PDB structure with molecular visualization tools"
            }

            return result, None
        elif response.status_code == 404:
            return None, f"Ligand not found: {ligand_id}"
        else:
            return None, f"Query failed: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="download_pdb",
            description="Download protein structure from RCSB PDB by PDB ID. USE FOR: Structure retrieval, docking preparation, structure analysis input. ENTITY TYPES: protein, structure. DATA FLOW: Produces PDB/mmCIF content for Rosetta/Vina/ProteinMPNN input.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_id": {
                        "type": "string",
                        "description": "4-letter PDB ID (e.g., '1TNR', '3ALQ')"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["pdb", "cif"],
                        "description": "Output format: 'pdb' or 'cif' (default: 'pdb')"
                    }
                },
                "required": ["pdb_id"]
            }
        ),
        Tool(
            name="search_structures",
            description="Search protein structures by text query (protein name, gene name, organism). USE FOR: Structure discovery, template identification, target structure finding. ENTITY TYPES: protein, gene, structure. DATA FLOW: Produces PDB IDs for download_pdb or analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'TNF receptor', 'IL-11', 'TNFRSF1A human')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_binding_sites",
            description="Analyze binding sites and ligands in a protein structure. USE FOR: Drug binding site identification, hotspot residue finding, docking target preparation. ENTITY TYPES: protein, compound, structure. DATA FLOW: Produces ligand positions and binding residues for binder design.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_id": {
                        "type": "string",
                        "description": "4-letter PDB ID"
                    }
                },
                "required": ["pdb_id"]
            }
        ),
        Tool(
            name="get_ligand_interactions",
            description="Get detailed ligand-protein interaction information. USE FOR: Interaction analysis, drug mechanism understanding, binding mode study. ENTITY TYPES: compound, protein. DATA FLOW: Produces interaction details for drug design optimization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_id": {
                        "type": "string",
                        "description": "4-letter PDB ID"
                    },
                    "ligand_id": {
                        "type": "string",
                        "description": "3-letter ligand ID (e.g., 'ATP', 'NAG')"
                    }
                },
                "required": ["pdb_id", "ligand_id"]
            }
        ),
        Tool(
            name="search_by_sequence",
            description="Search structures by protein sequence similarity. USE FOR: Template structure finding, homolog identification, structure prediction preparation. ENTITY TYPES: protein, sequence. DATA FLOW: Requires sequence, produces similar PDB structures for modeling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein sequence in single-letter amino acid code"
                    },
                    "identity_cutoff": {
                        "type": "number",
                        "description": "Minimum sequence identity (0-1, default: 0.9 = 90%)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)"
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="get_protein_info",
            description="Get detailed metadata about a protein structure. USE FOR: Structure quality assessment, citation retrieval, entity identification. ENTITY TYPES: protein, structure. DATA FLOW: Produces resolution, method, citation for structure selection criteria.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_id": {
                        "type": "string",
                        "description": "4-letter PDB ID"
                    }
                },
                "required": ["pdb_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "download_pdb":
        pdb_id = arguments.get("pdb_id", "")
        format = arguments.get("format", "pdb")
        content, error = download_structure(pdb_id, format)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=content)]

    elif name == "search_structures":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        results, error = search_by_text(query, max_results)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps({"query": query, "results": results, "count": len(results)}, indent=2))]

    elif name == "get_binding_sites":
        pdb_id = arguments.get("pdb_id", "")
        result, error = get_binding_sites(pdb_id)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_ligand_interactions":
        pdb_id = arguments.get("pdb_id", "")
        ligand_id = arguments.get("ligand_id", "")
        result, error = get_ligand_interactions(pdb_id, ligand_id)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "search_by_sequence":
        sequence = arguments.get("sequence", "")
        identity_cutoff = arguments.get("identity_cutoff", 0.9)
        max_results = arguments.get("max_results", 10)
        results, error = search_by_sequence(sequence, identity_cutoff, max_results)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps({"sequence_length": len(sequence), "identity_cutoff": identity_cutoff, "results": results, "count": len(results) if results else 0}, indent=2))]

    elif name == "get_protein_info":
        pdb_id = arguments.get("pdb_id", "")
        info, error = get_entry_info(pdb_id)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(info, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    logger.info("Starting RCSB PDB MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
