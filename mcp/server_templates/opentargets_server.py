"""
Open Targets MCP Server

Provides drug-target association and druggability analysis using Open Targets Platform API.
Alternative to DrugBank with free access.

Tools:
- search_target_drugs: Find drugs targeting a gene
- assess_druggability: Evaluate target druggability
- get_drug_info: Get drug details
- get_target_info: Get target information
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
logger = logging.getLogger("opentargets-mcp")

# Open Targets Platform API
OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"

# Create MCP server instance
app = Server("opentargets")


def graphql_query(query: str, variables: dict = None) -> tuple[Optional[Dict], Optional[str]]:
    """
    Execute GraphQL query against Open Targets API.

    Returns:
        (data, error_message)
    """
    try:
        response = requests.post(
            OPENTARGETS_API,
            json={"query": query, "variables": variables or {}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                return None, str(data["errors"])
            return data.get("data"), None
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def search_target(gene_symbol: str) -> tuple[Optional[str], Optional[str]]:
    """
    Search for target by gene symbol and return Ensembl ID.

    Returns:
        (ensembl_id, error_message)
    """
    query = """
    query SearchTarget($queryString: String!) {
        search(queryString: $queryString, entityNames: ["target"], page: {size: 5, index: 0}) {
            hits {
                id
                entity
                name
                description
            }
        }
    }
    """

    data, error = graphql_query(query, {"queryString": gene_symbol})
    if error:
        return None, error

    hits = data.get("search", {}).get("hits", [])
    if hits:
        # Return first target match
        for hit in hits:
            if hit.get("entity") == "target":
                return hit.get("id"), None

    return None, f"Target not found: {gene_symbol}"


def get_target_drugs(ensembl_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get drugs targeting a specific gene.

    Returns:
        (drug_data, error_message)
    """
    query = """
    query TargetDrugs($ensemblId: String!) {
        target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            knownDrugs {
                uniqueDrugs
                count
                rows {
                    drug {
                        id
                        name
                        drugType
                        maximumClinicalTrialPhase
                        hasBeenWithdrawn
                        mechanismsOfAction {
                            rows {
                                mechanismOfAction
                                actionType
                            }
                        }
                    }
                    phase
                    status
                    ctIds
                }
            }
        }
    }
    """

    data, error = graphql_query(query, {"ensemblId": ensembl_id})
    if error:
        return None, error

    target = data.get("target")
    if not target:
        return None, f"Target not found: {ensembl_id}"

    drugs = []
    known_drugs = target.get("knownDrugs", {})
    for row in known_drugs.get("rows", []):
        drug = row.get("drug", {})
        moa = drug.get("mechanismsOfAction", {}).get("rows", [])

        drugs.append({
            "drug_id": drug.get("id"),
            "drug_name": drug.get("name"),
            "drug_type": drug.get("drugType"),
            "max_phase": drug.get("maximumClinicalTrialPhase"),
            "withdrawn": drug.get("hasBeenWithdrawn"),
            "trial_phase": row.get("phase"),
            "trial_status": row.get("status"),
            "mechanism": moa[0].get("mechanismOfAction") if moa else None,
            "action_type": moa[0].get("actionType") if moa else None
        })

    return {
        "target_id": target.get("id"),
        "gene_symbol": target.get("approvedSymbol"),
        "gene_name": target.get("approvedName"),
        "total_drugs": known_drugs.get("uniqueDrugs", 0),
        "drugs": drugs
    }, None


def get_target_tractability(ensembl_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get tractability (druggability) assessment for a target.

    Returns:
        (tractability_data, error_message)
    """
    query = """
    query TargetTractability($ensemblId: String!) {
        target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            tractability {
                label
                modality
                value
            }
        }
    }
    """

    data, error = graphql_query(query, {"ensemblId": ensembl_id})
    if error:
        return None, error

    target = data.get("target")
    if not target:
        return None, f"Target not found: {ensembl_id}"

    tractability = {}
    for t in target.get("tractability", []):
        modality = t.get("modality", "unknown")
        if modality not in tractability:
            tractability[modality] = []
        tractability[modality].append({
            "label": t.get("label"),
            "value": t.get("value")
        })

    return {
        "target_id": target.get("id"),
        "gene_symbol": target.get("approvedSymbol"),
        "tractability": tractability,
        "summary": {
            "small_molecule": any(t.get("value") for t in tractability.get("SM", [])),
            "antibody": any(t.get("value") for t in tractability.get("AB", [])),
            "other_modalities": any(t.get("value") for t in tractability.get("OC", []))
        }
    }, None


def get_drug_details(chembl_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Get drug details by ChEMBL ID.

    Returns:
        (drug_data, error_message)
    """
    query = """
    query DrugInfo($chemblId: String!) {
        drug(chemblId: $chemblId) {
            id
            name
            drugType
            maximumClinicalTrialPhase
            hasBeenWithdrawn
            description
            synonyms
            tradeNames
            linkedTargets {
                rows {
                    id
                    approvedSymbol
                    approvedName
                }
            }
            linkedDiseases {
                rows {
                    id
                    name
                }
            }
        }
    }
    """

    data, error = graphql_query(query, {"chemblId": chembl_id})
    if error:
        return None, error

    drug = data.get("drug")
    if not drug:
        return None, f"Drug not found: {chembl_id}"

    return {
        "drug_id": drug.get("id"),
        "name": drug.get("name"),
        "type": drug.get("drugType"),
        "max_phase": drug.get("maximumClinicalTrialPhase"),
        "withdrawn": drug.get("hasBeenWithdrawn"),
        "description": drug.get("description"),
        "synonyms": drug.get("synonyms", []),
        "trade_names": drug.get("tradeNames", []),
        "targets": [
            {"id": t.get("id"), "symbol": t.get("approvedSymbol"), "name": t.get("approvedName")}
            for t in drug.get("linkedTargets", {}).get("rows", [])
        ],
        "indications": [
            {"id": d.get("id"), "name": d.get("name")}
            for d in drug.get("linkedDiseases", {}).get("rows", [])
        ]
    }, None


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_target_drugs",
            description="Find drugs targeting a specific gene with clinical data. USE FOR: Existing drug discovery, competitive landscape analysis, repurposing opportunities. ENTITY TYPES: gene, drug. DATA FLOW: Produces drug-target pairs for drug development strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Gene symbol (e.g., 'TNFRSF1A', 'IL11', 'EGFR')"
                    }
                },
                "required": ["gene_symbol"]
            }
        ),
        Tool(
            name="assess_druggability",
            description="Evaluate target druggability for different modalities. USE FOR: Target prioritization, modality selection, feasibility assessment. ENTITY TYPES: gene, protein. DATA FLOW: Produces tractability scores for small molecules, antibodies, and biologics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Gene symbol (e.g., 'TNFRSF1A', 'IL11')"
                    }
                },
                "required": ["gene_symbol"]
            }
        ),
        Tool(
            name="get_drug_info",
            description="Get detailed drug information with targets and indications. USE FOR: Drug mechanism analysis, indication study, synonym identification. ENTITY TYPES: drug, protein. DATA FLOW: Produces comprehensive drug profile for competitive analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chembl_id": {
                        "type": "string",
                        "description": "ChEMBL ID (e.g., 'CHEMBL1201580' for Infliximab)"
                    }
                },
                "required": ["chembl_id"]
            }
        ),
        Tool(
            name="get_target_info",
            description="Get drug target information with Ensembl ID. USE FOR: Target identification, gene-Ensembl mapping, target validation. ENTITY TYPES: gene, protein. DATA FLOW: Produces Ensembl IDs for cross-database target queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Gene symbol"
                    }
                },
                "required": ["gene_symbol"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "search_target_drugs":
        gene_symbol = arguments.get("gene_symbol", "")

        # First get Ensembl ID
        ensembl_id, error = search_target(gene_symbol)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]

        result, error = get_target_drugs(ensembl_id)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "assess_druggability":
        gene_symbol = arguments.get("gene_symbol", "")

        ensembl_id, error = search_target(gene_symbol)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]

        result, error = get_target_tractability(ensembl_id)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_drug_info":
        chembl_id = arguments.get("chembl_id", "")

        result, error = get_drug_details(chembl_id)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_target_info":
        gene_symbol = arguments.get("gene_symbol", "")

        ensembl_id, error = search_target(gene_symbol)
        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]

        return [TextContent(type="text", text=json.dumps({
            "gene_symbol": gene_symbol,
            "ensembl_id": ensembl_id
        }, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    logger.info("Starting Open Targets MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
