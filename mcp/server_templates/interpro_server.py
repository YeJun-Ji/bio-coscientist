"""
InterPro MCP Server

Provides protein domain and family analysis using the InterPro API.
Supports domain identification, architecture analysis, and cross-database queries.

Tools:
- analyze_domains: Analyze protein domains from UniProt ID
- get_entry_info: Get InterPro entry details
- get_domain_architecture: Get domain architecture for a protein
- search_by_domain: Find proteins with specific domain
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
logger = logging.getLogger("interpro-mcp")

# InterPro API endpoints
INTERPRO_API = "https://www.ebi.ac.uk/interpro/api"

# Create MCP server instance
app = Server("interpro")


def get_protein_domains(uniprot_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get domain information for a UniProt protein.

    Returns:
        (domain_data, error_message)
    """
    uniprot_id = uniprot_id.strip().upper()
    url = f"{INTERPRO_API}/protein/UniProt/{uniprot_id}"

    try:
        logger.info(f"Fetching domains for {uniprot_id}")
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data, None
        elif response.status_code == 404:
            return None, f"UniProt ID not found: {uniprot_id}"
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def get_entry_details(interpro_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get details for an InterPro entry.

    Returns:
        (entry_data, error_message)
    """
    interpro_id = interpro_id.strip().upper()
    url = f"{INTERPRO_API}/entry/interpro/{interpro_id}"

    try:
        logger.info(f"Fetching entry {interpro_id}")
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=30)

        if response.status_code == 200:
            data = response.json()

            result = {
                "accession": data.get("metadata", {}).get("accession"),
                "name": data.get("metadata", {}).get("name", {}).get("name"),
                "short_name": data.get("metadata", {}).get("name", {}).get("short"),
                "type": data.get("metadata", {}).get("type"),
                "description": None,
                "go_terms": [],
                "member_databases": []
            }

            # Get description
            descriptions = data.get("metadata", {}).get("description", [])
            if descriptions:
                result["description"] = descriptions[0].get("text", "")

            # Get GO terms
            go_terms = data.get("metadata", {}).get("go_terms", [])
            for go in go_terms:
                result["go_terms"].append({
                    "id": go.get("identifier"),
                    "name": go.get("name"),
                    "category": go.get("category", {}).get("name")
                })

            # Get member databases
            member_dbs = data.get("metadata", {}).get("member_databases", {})
            for db_name, entries in member_dbs.items():
                for entry_id, entry_info in entries.items():
                    result["member_databases"].append({
                        "database": db_name,
                        "id": entry_id,
                        "name": entry_info.get("name") if isinstance(entry_info, dict) else None
                    })

            return result, None
        elif response.status_code == 404:
            return None, f"InterPro entry not found: {interpro_id}"
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def get_protein_entries(uniprot_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get all InterPro entries (domains/families) for a protein.

    Returns:
        (entries_data, error_message)
    """
    uniprot_id = uniprot_id.strip().upper()
    url = f"{INTERPRO_API}/protein/UniProt/{uniprot_id}/entry/interpro"

    try:
        logger.info(f"Fetching InterPro entries for {uniprot_id}")
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=30)

        if response.status_code == 200:
            data = response.json()

            domains = []
            for result in data.get("results", []):
                metadata = result.get("metadata", {})
                proteins = result.get("proteins", [])

                # Get location info from proteins
                locations = []
                for protein in proteins:
                    for entry_loc in protein.get("entry_protein_locations", []):
                        for fragment in entry_loc.get("fragments", []):
                            locations.append({
                                "start": fragment.get("start"),
                                "end": fragment.get("end")
                            })

                domain = {
                    "accession": metadata.get("accession"),
                    "name": metadata.get("name", {}).get("name") if isinstance(metadata.get("name"), dict) else metadata.get("name"),
                    "type": metadata.get("type"),
                    "locations": locations
                }
                domains.append(domain)

            # Sort by start position
            domains.sort(key=lambda x: x["locations"][0]["start"] if x["locations"] else 0)

            return {
                "uniprot_id": uniprot_id,
                "domains": domains,
                "num_domains": len(domains)
            }, None
        elif response.status_code == 404:
            return None, f"UniProt ID not found: {uniprot_id}"
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def search_proteins_by_entry(interpro_id: str, max_results: int = 20) -> tuple[Optional[List], Optional[str]]:
    """
    Find proteins containing a specific InterPro entry.

    Returns:
        (proteins, error_message)
    """
    interpro_id = interpro_id.strip().upper()
    url = f"{INTERPRO_API}/protein/UniProt/entry/interpro/{interpro_id}"

    try:
        logger.info(f"Searching proteins with {interpro_id}")
        response = requests.get(
            url,
            params={"page_size": max_results},
            headers={"Accept": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            proteins = []
            for result in data.get("results", []):
                metadata = result.get("metadata", {})
                proteins.append({
                    "accession": metadata.get("accession"),
                    "name": metadata.get("name"),
                    "organism": metadata.get("source_organism", {}).get("scientificName") if metadata.get("source_organism") else None,
                    "length": metadata.get("length")
                })

            return {
                "interpro_id": interpro_id,
                "proteins": proteins,
                "count": len(proteins)
            }, None
        elif response.status_code == 404:
            return None, f"InterPro entry not found: {interpro_id}"
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="analyze_domains",
            description="Analyze protein domains for a UniProt protein. USE FOR: Domain identification, functional annotation, protein family classification. ENTITY TYPES: protein, domain. DATA FLOW: Requires UniProt ID, produces domain annotations for target characterization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "uniprot_id": {
                        "type": "string",
                        "description": "UniProt accession ID (e.g., 'P19438' for TNFR1, 'P22301' for IL-10)"
                    }
                },
                "required": ["uniprot_id"]
            }
        ),
        Tool(
            name="get_entry_info",
            description="Get detailed InterPro entry information with GO terms. USE FOR: Domain function understanding, GO term retrieval, cross-database links. ENTITY TYPES: domain, pathway. DATA FLOW: Produces GO terms and member database entries for functional analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "interpro_id": {
                        "type": "string",
                        "description": "InterPro accession (e.g., 'IPR000488' for Death domain)"
                    }
                },
                "required": ["interpro_id"]
            }
        ),
        Tool(
            name="get_domain_architecture",
            description="Get domain architecture with ordered positions. USE FOR: Protein organization analysis, binding site localization, domain boundary identification. ENTITY TYPES: protein, domain. DATA FLOW: Produces domain order and positions for binder design targeting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "uniprot_id": {
                        "type": "string",
                        "description": "UniProt accession ID"
                    }
                },
                "required": ["uniprot_id"]
            }
        ),
        Tool(
            name="search_by_domain",
            description="Find proteins containing a specific domain or family. USE FOR: Homolog discovery, family member identification, off-target assessment. ENTITY TYPES: protein, domain. DATA FLOW: Produces protein list for selectivity and specificity analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "interpro_id": {
                        "type": "string",
                        "description": "InterPro accession to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)"
                    }
                },
                "required": ["interpro_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "analyze_domains":
        uniprot_id = arguments.get("uniprot_id", "")
        result, error = get_protein_entries(uniprot_id)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_entry_info":
        interpro_id = arguments.get("interpro_id", "")
        result, error = get_entry_details(interpro_id)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_domain_architecture":
        uniprot_id = arguments.get("uniprot_id", "")
        result, error = get_protein_entries(uniprot_id)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]

        # Format as visual architecture
        architecture = {
            "uniprot_id": uniprot_id,
            "domains": result.get("domains", []),
            "architecture_diagram": ""
        }

        # Create simple text diagram
        if result.get("domains"):
            domains = result["domains"]
            max_pos = max(loc["end"] for d in domains for loc in d.get("locations", []) if loc.get("end"))

            diagram_parts = []
            for d in domains:
                for loc in d.get("locations", []):
                    start = loc.get("start", 0)
                    end = loc.get("end", 0)
                    diagram_parts.append(f"[{start}-{end}] {d.get('name', d.get('accession'))}")

            architecture["architecture_diagram"] = " | ".join(diagram_parts)
            architecture["protein_length_estimate"] = max_pos

        return [TextContent(type="text", text=json.dumps(architecture, indent=2))]

    elif name == "search_by_domain":
        interpro_id = arguments.get("interpro_id", "")
        max_results = arguments.get("max_results", 20)
        result, error = search_proteins_by_entry(interpro_id, max_results)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    logger.info("Starting InterPro MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
