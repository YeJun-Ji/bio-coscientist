"""
STRING DB MCP Server

Provides protein-protein interaction network analysis using STRING database API.

Tools:
- get_protein_network: Get interaction network for a protein
- get_interaction_partners: Get interaction partners with confidence scores
- get_enrichment_analysis: Perform functional enrichment analysis
- get_network_image: Get network visualization image
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
logger = logging.getLogger("stringdb-mcp")

# STRING DB API endpoints
STRING_API_BASE = "https://string-db.org/api"
STRING_VERSION = "12.0"

# Default settings
DEFAULT_SPECIES = 9606  # Homo sapiens
DEFAULT_SCORE_THRESHOLD = 400  # Medium confidence

# Create MCP server instance
app = Server("stringdb")


def resolve_protein_id(protein_name: str, species: int = DEFAULT_SPECIES) -> Optional[str]:
    """
    Resolve protein name to STRING ID.

    Args:
        protein_name: Protein name or gene symbol
        species: NCBI taxonomy ID (default: 9606 for human)

    Returns:
        STRING protein ID or None
    """
    url = f"{STRING_API_BASE}/json/get_string_ids"
    params = {
        "identifiers": protein_name,
        "species": species,
        "limit": 1,
        "caller_identity": "biocoscientist"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            return data[0].get("stringId")
        return None
    except Exception as e:
        logger.error(f"Failed to resolve protein ID: {e}")
        return None


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available STRING DB tools."""
    return [
        Tool(
            name="get_protein_network",
            description="Get protein-protein interaction network from STRING database. USE FOR: PPI network construction, interactor discovery, pathway mapping. ENTITY TYPES: protein, gene. DATA FLOW: Produces interaction edges with confidence scores for NetworkX analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "protein_id": {
                        "type": "string",
                        "description": "Protein name or STRING ID (e.g., 'IL11', 'TP53', '9606.ENSP00000269305')"
                    },
                    "species": {
                        "type": "integer",
                        "description": "NCBI taxonomy ID (default: 9606 for human)",
                        "default": 9606
                    },
                    "score_threshold": {
                        "type": "integer",
                        "description": "Minimum interaction score (0-1000, default: 400 for medium confidence)",
                        "default": 400
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of interactions to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["protein_id"]
            }
        ),
        Tool(
            name="get_interaction_partners",
            description="Get interaction partners for a protein with confidence scores. USE FOR: Interactor identification, hub protein discovery. ENTITY TYPES: protein, gene. DATA FLOW: Produces partner list for network analysis or enrichment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "protein_id": {
                        "type": "string",
                        "description": "Protein name or STRING ID"
                    },
                    "species": {
                        "type": "integer",
                        "description": "NCBI taxonomy ID (default: 9606 for human)",
                        "default": 9606
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of partners (default: 10)",
                        "default": 10
                    }
                },
                "required": ["protein_id"]
            }
        ),
        Tool(
            name="get_enrichment_analysis",
            description="Perform GO/KEGG functional enrichment on protein list. USE FOR: Functional annotation, pathway enrichment, biological process identification. ENTITY TYPES: protein, gene, pathway. DATA FLOW: Requires protein list, produces enriched terms and pathways.",
            inputSchema={
                "type": "object",
                "properties": {
                    "protein_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of protein names or STRING IDs"
                    },
                    "species": {
                        "type": "integer",
                        "description": "NCBI taxonomy ID (default: 9606 for human)",
                        "default": 9606
                    }
                },
                "required": ["protein_list"]
            }
        ),
        Tool(
            name="get_network_image",
            description="Get network visualization image URL. USE FOR: Network visualization, figure generation. ENTITY TYPES: protein. DATA FLOW: Requires protein list, produces image URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "protein_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of protein names or STRING IDs to visualize"
                    },
                    "species": {
                        "type": "integer",
                        "description": "NCBI taxonomy ID (default: 9606 for human)",
                        "default": 9606
                    }
                },
                "required": ["protein_list"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "get_protein_network":
            result = await get_protein_network(
                protein_id=arguments["protein_id"],
                species=arguments.get("species", DEFAULT_SPECIES),
                score_threshold=arguments.get("score_threshold", DEFAULT_SCORE_THRESHOLD),
                limit=arguments.get("limit", 20)
            )
        elif name == "get_interaction_partners":
            result = await get_interaction_partners(
                protein_id=arguments["protein_id"],
                species=arguments.get("species", DEFAULT_SPECIES),
                limit=arguments.get("limit", 10)
            )
        elif name == "get_enrichment_analysis":
            result = await get_enrichment_analysis(
                protein_list=arguments["protein_list"],
                species=arguments.get("species", DEFAULT_SPECIES)
            )
        elif name == "get_network_image":
            result = await get_network_image(
                protein_list=arguments["protein_list"],
                species=arguments.get("species", DEFAULT_SPECIES)
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def get_protein_network(
    protein_id: str,
    species: int = DEFAULT_SPECIES,
    score_threshold: int = DEFAULT_SCORE_THRESHOLD,
    limit: int = 20
) -> Dict[str, Any]:
    """Get protein-protein interaction network."""

    # Resolve protein name to STRING ID if needed
    string_id = protein_id
    if not protein_id.startswith(str(species)):
        resolved = resolve_protein_id(protein_id, species)
        if resolved:
            string_id = resolved
        else:
            return {"error": f"Could not resolve protein: {protein_id}"}

    url = f"{STRING_API_BASE}/json/network"
    params = {
        "identifiers": string_id,
        "species": species,
        "required_score": score_threshold,
        "limit": limit,
        "caller_identity": "biocoscientist"
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        interactions = response.json()

        # Process and format results
        network = {
            "query_protein": protein_id,
            "string_id": string_id,
            "species": species,
            "score_threshold": score_threshold,
            "interaction_count": len(interactions),
            "interactions": []
        }

        for interaction in interactions[:limit]:
            network["interactions"].append({
                "protein_a": interaction.get("preferredName_A", interaction.get("stringId_A")),
                "protein_b": interaction.get("preferredName_B", interaction.get("stringId_B")),
                "combined_score": interaction.get("score", 0),
                "experimental_score": interaction.get("escore", 0),
                "database_score": interaction.get("dscore", 0),
                "textmining_score": interaction.get("tscore", 0),
                "coexpression_score": interaction.get("ascore", 0)
            })

        return network

    except Exception as e:
        logger.error(f"Failed to get network: {e}")
        return {"error": str(e)}


async def get_interaction_partners(
    protein_id: str,
    species: int = DEFAULT_SPECIES,
    limit: int = 10
) -> Dict[str, Any]:
    """Get interaction partners with confidence scores."""

    # Resolve protein name to STRING ID if needed
    string_id = protein_id
    if not protein_id.startswith(str(species)):
        resolved = resolve_protein_id(protein_id, species)
        if resolved:
            string_id = resolved
        else:
            return {"error": f"Could not resolve protein: {protein_id}"}

    url = f"{STRING_API_BASE}/json/interaction_partners"
    params = {
        "identifiers": string_id,
        "species": species,
        "limit": limit,
        "caller_identity": "biocoscientist"
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        partners = response.json()

        result = {
            "query_protein": protein_id,
            "string_id": string_id,
            "partner_count": len(partners),
            "partners": []
        }

        for partner in partners:
            result["partners"].append({
                "protein": partner.get("preferredName_B", partner.get("stringId_B")),
                "string_id": partner.get("stringId_B"),
                "combined_score": partner.get("score", 0),
                "annotation": partner.get("annotation_B", "")
            })

        return result

    except Exception as e:
        logger.error(f"Failed to get partners: {e}")
        return {"error": str(e)}


async def get_enrichment_analysis(
    protein_list: List[str],
    species: int = DEFAULT_SPECIES
) -> Dict[str, Any]:
    """Perform functional enrichment analysis."""

    url = f"{STRING_API_BASE}/json/enrichment"
    params = {
        "identifiers": "\r".join(protein_list),
        "species": species,
        "caller_identity": "biocoscientist"
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        enrichments = response.json()

        # Group by category
        result = {
            "query_proteins": protein_list,
            "species": species,
            "enrichment_count": len(enrichments),
            "categories": {
                "GO_Biological_Process": [],
                "GO_Molecular_Function": [],
                "GO_Cellular_Component": [],
                "KEGG_Pathway": [],
                "Other": []
            }
        }

        for item in enrichments:
            category = item.get("category", "Other")
            entry = {
                "term": item.get("term", ""),
                "description": item.get("description", ""),
                "p_value": item.get("p_value", 1.0),
                "fdr": item.get("fdr", 1.0),
                "gene_count": item.get("number_of_genes", 0),
                "genes": item.get("inputGenes", "").split(",") if item.get("inputGenes") else []
            }

            if "Process" in category:
                result["categories"]["GO_Biological_Process"].append(entry)
            elif "Function" in category:
                result["categories"]["GO_Molecular_Function"].append(entry)
            elif "Component" in category:
                result["categories"]["GO_Cellular_Component"].append(entry)
            elif "KEGG" in category:
                result["categories"]["KEGG_Pathway"].append(entry)
            else:
                result["categories"]["Other"].append(entry)

        return result

    except Exception as e:
        logger.error(f"Failed to get enrichment: {e}")
        return {"error": str(e)}


async def get_network_image(
    protein_list: List[str],
    species: int = DEFAULT_SPECIES
) -> Dict[str, Any]:
    """Get network visualization image URL."""

    # Generate image URL
    proteins_str = "%0d".join(protein_list)
    image_url = (
        f"{STRING_API_BASE}/image/network?"
        f"identifiers={proteins_str}&"
        f"species={species}&"
        f"network_flavor=confidence&"
        f"caller_identity=biocoscientist"
    )

    return {
        "query_proteins": protein_list,
        "species": species,
        "image_url": image_url,
        "note": "Use this URL to view or download the network image"
    }


async def main():
    """Run the MCP server."""
    logger.info("Starting STRING DB MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
