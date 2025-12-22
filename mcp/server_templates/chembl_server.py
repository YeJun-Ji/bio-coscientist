"""
ChEMBL MCP Server

Provides drug-target interaction data and compound information from ChEMBL database.

Tools:
- search_target: Search for drug targets by name
- get_target_info: Get detailed target information
- search_compound: Search for compounds/drugs by name
- get_compound_activities: Get bioactivity data for a compound
- get_drug_mechanisms: Get drug mechanism of action
- search_approved_drugs: Find approved drugs targeting a protein
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
logger = logging.getLogger("chembl-mcp")

# ChEMBL API endpoints
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Create MCP server instance
app = Server("chembl")


def make_request(endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
    """Make a request to ChEMBL API with JSON format."""
    url = f"{CHEMBL_API_BASE}/{endpoint}"

    if params is None:
        params = {}
    params["format"] = "json"

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"ChEMBL API error: {e}")
        return None


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available ChEMBL tools."""
    return [
        Tool(
            name="search_target",
            description="Search drug targets by name in ChEMBL. USE FOR: Target identification, drug-target discovery, protein druggability check. ENTITY TYPES: protein, gene, drug. DATA FLOW: Produces ChEMBL target IDs for bioactivity queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "Target name or gene symbol (e.g., 'IL11', 'EGFR', 'TP53')"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism filter (default: 'Homo sapiens')",
                        "default": "Homo sapiens"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["target_name"]
            }
        ),
        Tool(
            name="get_target_info",
            description="Get detailed ChEMBL target information. USE FOR: Target characterization, UniProt cross-reference. ENTITY TYPES: protein, gene. DATA FLOW: Produces target details and UniProt accession for integration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_chembl_id": {
                        "type": "string",
                        "description": "ChEMBL target ID (e.g., 'CHEMBL2111342')"
                    }
                },
                "required": ["target_chembl_id"]
            }
        ),
        Tool(
            name="search_compound",
            description="Search compounds/drugs by name in ChEMBL. USE FOR: Drug lookup, compound identification, approval status check. ENTITY TYPES: compound, drug. DATA FLOW: Produces ChEMBL molecule IDs for activity queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "compound_name": {
                        "type": "string",
                        "description": "Compound or drug name (e.g., 'aspirin', 'imatinib')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["compound_name"]
            }
        ),
        Tool(
            name="get_compound_activities",
            description="Get IC50/Ki/Kd bioactivity data for compound. USE FOR: Potency assessment, target profiling, selectivity analysis. ENTITY TYPES: compound, protein. DATA FLOW: Produces activity values for efficacy evaluation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "molecule_chembl_id": {
                        "type": "string",
                        "description": "ChEMBL molecule ID (e.g., 'CHEMBL25')"
                    },
                    "activity_type": {
                        "type": "string",
                        "description": "Filter by activity type (e.g., 'IC50', 'Ki', 'Kd')",
                        "enum": ["IC50", "Ki", "Kd", "EC50", "all"],
                        "default": "all"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)",
                        "default": 20
                    }
                },
                "required": ["molecule_chembl_id"]
            }
        ),
        Tool(
            name="get_drug_mechanisms",
            description="Get drug mechanism of action information. USE FOR: Mechanism understanding, action type identification, target validation. ENTITY TYPES: drug, protein. DATA FLOW: Produces mechanism data (inhibitor/agonist/etc.) for drug design.",
            inputSchema={
                "type": "object",
                "properties": {
                    "molecule_chembl_id": {
                        "type": "string",
                        "description": "ChEMBL molecule ID (e.g., 'CHEMBL25')"
                    }
                },
                "required": ["molecule_chembl_id"]
            }
        ),
        Tool(
            name="search_approved_drugs",
            description="Find approved drugs targeting a protein. USE FOR: Drug repurposing, competitive landscape, existing therapy identification. ENTITY TYPES: protein, drug. DATA FLOW: Produces approved drug list for target validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_chembl_id": {
                        "type": "string",
                        "description": "ChEMBL target ID"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)",
                        "default": 20
                    }
                },
                "required": ["target_chembl_id"]
            }
        ),
        Tool(
            name="get_target_bioactivities",
            description="Get bioactivity data for compounds against target. USE FOR: Lead discovery, SAR analysis, hit identification. ENTITY TYPES: protein, compound. DATA FLOW: Produces ranked potency data sorted by pChEMBL value.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_chembl_id": {
                        "type": "string",
                        "description": "ChEMBL target ID"
                    },
                    "activity_type": {
                        "type": "string",
                        "description": "Filter by activity type",
                        "enum": ["IC50", "Ki", "Kd", "EC50", "all"],
                        "default": "all"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50)",
                        "default": 50
                    }
                },
                "required": ["target_chembl_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "search_target":
            result = await search_target(
                target_name=arguments["target_name"],
                organism=arguments.get("organism", "Homo sapiens"),
                limit=arguments.get("limit", 10)
            )
        elif name == "get_target_info":
            result = await get_target_info(
                target_chembl_id=arguments["target_chembl_id"]
            )
        elif name == "search_compound":
            result = await search_compound(
                compound_name=arguments["compound_name"],
                limit=arguments.get("limit", 10)
            )
        elif name == "get_compound_activities":
            result = await get_compound_activities(
                molecule_chembl_id=arguments["molecule_chembl_id"],
                activity_type=arguments.get("activity_type", "all"),
                limit=arguments.get("limit", 20)
            )
        elif name == "get_drug_mechanisms":
            result = await get_drug_mechanisms(
                molecule_chembl_id=arguments["molecule_chembl_id"]
            )
        elif name == "search_approved_drugs":
            result = await search_approved_drugs(
                target_chembl_id=arguments["target_chembl_id"],
                limit=arguments.get("limit", 20)
            )
        elif name == "get_target_bioactivities":
            result = await get_target_bioactivities(
                target_chembl_id=arguments["target_chembl_id"],
                activity_type=arguments.get("activity_type", "all"),
                limit=arguments.get("limit", 50)
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def search_target(
    target_name: str,
    organism: str = "Homo sapiens",
    limit: int = 10
) -> Dict[str, Any]:
    """Search for drug targets by name."""

    # Search targets
    data = make_request("target/search", {
        "q": target_name,
        "limit": limit
    })

    if not data or "targets" not in data:
        return {"error": f"No targets found for: {target_name}", "results": []}

    results = []
    for target in data.get("targets", []):
        # Filter by organism if specified
        if organism and target.get("organism") != organism:
            continue

        results.append({
            "target_chembl_id": target.get("target_chembl_id"),
            "pref_name": target.get("pref_name"),
            "target_type": target.get("target_type"),
            "organism": target.get("organism"),
            "target_components": [
                {
                    "component_type": comp.get("component_type"),
                    "accession": comp.get("accession")  # UniProt ID
                }
                for comp in target.get("target_components", [])
            ]
        })

    return {
        "query": target_name,
        "organism_filter": organism,
        "result_count": len(results),
        "targets": results[:limit]
    }


async def get_target_info(target_chembl_id: str) -> Dict[str, Any]:
    """Get detailed target information."""

    data = make_request(f"target/{target_chembl_id}")

    if not data:
        return {"error": f"Target not found: {target_chembl_id}"}

    return {
        "target_chembl_id": data.get("target_chembl_id"),
        "pref_name": data.get("pref_name"),
        "target_type": data.get("target_type"),
        "organism": data.get("organism"),
        "species_group_flag": data.get("species_group_flag"),
        "target_components": [
            {
                "component_type": comp.get("component_type"),
                "accession": comp.get("accession"),
                "component_description": comp.get("component_description")
            }
            for comp in data.get("target_components", [])
        ],
        "cross_references": data.get("cross_references", [])
    }


async def search_compound(
    compound_name: str,
    limit: int = 10
) -> Dict[str, Any]:
    """Search for compounds by name."""

    data = make_request("molecule/search", {
        "q": compound_name,
        "limit": limit
    })

    if not data or "molecules" not in data:
        return {"error": f"No compounds found for: {compound_name}", "results": []}

    results = []
    for mol in data.get("molecules", []):
        results.append({
            "molecule_chembl_id": mol.get("molecule_chembl_id"),
            "pref_name": mol.get("pref_name"),
            "molecule_type": mol.get("molecule_type"),
            "max_phase": mol.get("max_phase"),  # 4 = approved drug
            "first_approval": mol.get("first_approval"),
            "oral": mol.get("oral"),
            "parenteral": mol.get("parenteral"),
            "topical": mol.get("topical"),
            "molecule_properties": {
                "mw_freebase": mol.get("molecule_properties", {}).get("mw_freebase"),
                "alogp": mol.get("molecule_properties", {}).get("alogp"),
                "hba": mol.get("molecule_properties", {}).get("hba"),
                "hbd": mol.get("molecule_properties", {}).get("hbd"),
                "psa": mol.get("molecule_properties", {}).get("psa"),
                "ro3_pass": mol.get("molecule_properties", {}).get("ro3_pass"),
                "num_ro5_violations": mol.get("molecule_properties", {}).get("num_ro5_violations")
            } if mol.get("molecule_properties") else None
        })

    return {
        "query": compound_name,
        "result_count": len(results),
        "compounds": results
    }


async def get_compound_activities(
    molecule_chembl_id: str,
    activity_type: str = "all",
    limit: int = 20
) -> Dict[str, Any]:
    """Get bioactivity data for a compound."""

    params = {
        "molecule_chembl_id": molecule_chembl_id,
        "limit": limit
    }

    if activity_type != "all":
        params["standard_type"] = activity_type

    data = make_request("activity", params)

    if not data or "activities" not in data:
        return {"error": f"No activities found for: {molecule_chembl_id}", "results": []}

    activities = []
    for act in data.get("activities", []):
        activities.append({
            "activity_id": act.get("activity_id"),
            "target_chembl_id": act.get("target_chembl_id"),
            "target_pref_name": act.get("target_pref_name"),
            "target_organism": act.get("target_organism"),
            "assay_type": act.get("assay_type"),
            "standard_type": act.get("standard_type"),  # IC50, Ki, etc.
            "standard_value": act.get("standard_value"),
            "standard_units": act.get("standard_units"),
            "standard_relation": act.get("standard_relation"),  # =, <, >, etc.
            "pchembl_value": act.get("pchembl_value"),  # -log10 normalized
            "data_validity_comment": act.get("data_validity_comment")
        })

    return {
        "molecule_chembl_id": molecule_chembl_id,
        "activity_type_filter": activity_type,
        "activity_count": len(activities),
        "activities": activities
    }


async def get_drug_mechanisms(molecule_chembl_id: str) -> Dict[str, Any]:
    """Get mechanism of action for a drug."""

    data = make_request("mechanism", {
        "molecule_chembl_id": molecule_chembl_id
    })

    if not data or "mechanisms" not in data:
        return {"molecule_chembl_id": molecule_chembl_id, "mechanisms": []}

    mechanisms = []
    for mech in data.get("mechanisms", []):
        mechanisms.append({
            "mechanism_of_action": mech.get("mechanism_of_action"),
            "action_type": mech.get("action_type"),  # INHIBITOR, AGONIST, etc.
            "target_chembl_id": mech.get("target_chembl_id"),
            "target_name": mech.get("target_name"),
            "disease_efficacy": mech.get("disease_efficacy"),
            "selectivity_comment": mech.get("selectivity_comment"),
            "binding_site_comment": mech.get("binding_site_comment")
        })

    return {
        "molecule_chembl_id": molecule_chembl_id,
        "mechanism_count": len(mechanisms),
        "mechanisms": mechanisms
    }


async def search_approved_drugs(
    target_chembl_id: str,
    limit: int = 20
) -> Dict[str, Any]:
    """Find approved drugs targeting a protein."""

    # Get activities for the target with approved drugs (max_phase = 4)
    data = make_request("activity", {
        "target_chembl_id": target_chembl_id,
        "limit": 1000  # Get more to filter
    })

    if not data or "activities" not in data:
        return {"target_chembl_id": target_chembl_id, "approved_drugs": []}

    # Collect unique molecules
    molecule_ids = set()
    for act in data.get("activities", []):
        mol_id = act.get("molecule_chembl_id")
        if mol_id:
            molecule_ids.add(mol_id)

    # Check each molecule for approval status
    approved_drugs = []
    for mol_id in list(molecule_ids)[:50]:  # Limit API calls
        mol_data = make_request(f"molecule/{mol_id}")
        if mol_data and mol_data.get("max_phase") == 4:
            approved_drugs.append({
                "molecule_chembl_id": mol_data.get("molecule_chembl_id"),
                "pref_name": mol_data.get("pref_name"),
                "first_approval": mol_data.get("first_approval"),
                "oral": mol_data.get("oral"),
                "indication_class": mol_data.get("indication_class")
            })

        if len(approved_drugs) >= limit:
            break

    return {
        "target_chembl_id": target_chembl_id,
        "approved_drug_count": len(approved_drugs),
        "approved_drugs": approved_drugs
    }


async def get_target_bioactivities(
    target_chembl_id: str,
    activity_type: str = "all",
    limit: int = 50
) -> Dict[str, Any]:
    """Get bioactivity data for compounds tested against a target."""

    params = {
        "target_chembl_id": target_chembl_id,
        "limit": limit
    }

    if activity_type != "all":
        params["standard_type"] = activity_type

    data = make_request("activity", params)

    if not data or "activities" not in data:
        return {"error": f"No activities found for target: {target_chembl_id}"}

    activities = []
    for act in data.get("activities", []):
        activities.append({
            "molecule_chembl_id": act.get("molecule_chembl_id"),
            "molecule_pref_name": act.get("molecule_pref_name"),
            "standard_type": act.get("standard_type"),
            "standard_value": act.get("standard_value"),
            "standard_units": act.get("standard_units"),
            "pchembl_value": act.get("pchembl_value"),
            "assay_description": act.get("assay_description")
        })

    # Sort by pchembl_value (higher = more potent)
    activities.sort(
        key=lambda x: float(x.get("pchembl_value") or 0),
        reverse=True
    )

    return {
        "target_chembl_id": target_chembl_id,
        "activity_type_filter": activity_type,
        "activity_count": len(activities),
        "activities": activities,
        "note": "Sorted by pchembl_value (higher = more potent)"
    }


async def main():
    """Run the MCP server."""
    logger.info("Starting ChEMBL MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
