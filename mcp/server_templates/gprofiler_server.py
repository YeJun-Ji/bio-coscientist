"""
gProfiler MCP Server

Provides functional enrichment analysis using the g:Profiler API.
Supports GO, KEGG, Reactome enrichment and gene ID conversion.

Tools:
- enrichment_analysis: GO/KEGG/Reactome enrichment
- convert_gene_ids: Gene ID conversion
- compare_gene_lists: Multi-query comparison
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
logger = logging.getLogger("gprofiler-mcp")

# g:Profiler API endpoint
GPROFILER_API = "https://biit.cs.ut.ee/gprofiler/api"

# Create MCP server instance
app = Server("gprofiler")


def run_enrichment(
    genes: List[str],
    organism: str = "hsapiens",
    sources: List[str] = None,
    significance_threshold: float = 0.05
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Run functional enrichment analysis.

    Args:
        genes: List of gene symbols or IDs
        organism: Organism code (default: hsapiens)
        sources: Data sources (GO:BP, GO:MF, GO:CC, KEGG, REAC, etc.)
        significance_threshold: P-value threshold

    Returns:
        (results, error_message)
    """
    if sources is None:
        sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]

    url = f"{GPROFILER_API}/gost/profile/"

    payload = {
        "organism": organism,
        "query": genes,
        "sources": sources,
        "user_threshold": significance_threshold,
        "all_results": False,
        "ordered": False,
        "combined": False,
        "measure_underrepresentation": False,
        "no_iea": False,
        "domain_scope": "annotated",
        "numeric_namespace": "ENTREZGENE_ACC",
        "significance_threshold_method": "g_SCS"
    }

    try:
        logger.info(f"Running enrichment for {len(genes)} genes")
        response = requests.post(url, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()

            results = []
            for result in data.get("result", []):
                enriched = {
                    "source": result.get("source"),
                    "term_id": result.get("native"),
                    "term_name": result.get("name"),
                    "p_value": result.get("p_value"),
                    "term_size": result.get("term_size"),
                    "query_size": result.get("query_size"),
                    "intersection_size": result.get("intersection_size"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                    "intersections": result.get("intersections", [])
                }
                results.append(enriched)

            # Sort by p-value
            results.sort(key=lambda x: x["p_value"])

            return {
                "query_genes": len(genes),
                "organism": organism,
                "sources": sources,
                "significant_terms": len(results),
                "results": results
            }, None
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def convert_ids(
    genes: List[str],
    organism: str = "hsapiens",
    target_namespace: str = "ENSG"
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Convert gene IDs between namespaces.

    Args:
        genes: List of gene IDs
        organism: Organism code
        target_namespace: Target namespace (ENSG, UNIPROT, ENTREZGENE, etc.)

    Returns:
        (conversion_results, error_message)
    """
    url = f"{GPROFILER_API}/convert/convert/"

    payload = {
        "organism": organism,
        "query": genes,
        "target": target_namespace,
        "numeric_namespace": "ENTREZGENE_ACC"
    }

    try:
        logger.info(f"Converting {len(genes)} gene IDs to {target_namespace}")
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()

            conversions = []
            for result in data.get("result", []):
                conv = {
                    "input": result.get("incoming"),
                    "output": result.get("converted"),
                    "name": result.get("name"),
                    "description": result.get("description"),
                    "namespace": result.get("namespaces")
                }
                conversions.append(conv)

            return {
                "input_count": len(genes),
                "converted_count": len([c for c in conversions if c["output"]]),
                "target_namespace": target_namespace,
                "conversions": conversions
            }, None
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def get_gene_info(gene: str, organism: str = "hsapiens") -> tuple[Optional[Dict], Optional[str]]:
    """
    Get gene information.

    Returns:
        (gene_info, error_message)
    """
    url = f"{GPROFILER_API}/convert/convert/"

    payload = {
        "organism": organism,
        "query": [gene],
        "target": "ENSG"
    }

    try:
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            results = data.get("result", [])

            if results:
                result = results[0]
                return {
                    "query": gene,
                    "ensembl_id": result.get("converted"),
                    "name": result.get("name"),
                    "description": result.get("description"),
                    "namespaces": result.get("namespaces")
                }, None
            else:
                return None, f"Gene not found: {gene}"
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="enrichment_analysis",
            description="Run GO/KEGG/Reactome functional enrichment analysis. USE FOR: Pathway identification, biological process discovery, functional clustering. ENTITY TYPES: gene, pathway. DATA FLOW: Requires gene list, produces enriched terms for target validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene symbols or IDs (e.g., ['IL11', 'STAT3', 'JAK1'])"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism code (default: 'hsapiens' for human, 'mmusculus' for mouse)"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Data sources: GO:BP, GO:MF, GO:CC, KEGG, REAC (default: all)"
                    },
                    "significance_threshold": {
                        "type": "number",
                        "description": "P-value threshold (default: 0.05)"
                    }
                },
                "required": ["genes"]
            }
        ),
        Tool(
            name="convert_gene_ids",
            description="Convert gene IDs between namespaces (ENSG, UNIPROT, ENTREZGENE). USE FOR: ID mapping, cross-database queries, data integration. ENTITY TYPES: gene, protein. DATA FLOW: Enables ID conversion for multi-database pipeline integration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene IDs to convert"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism code (default: 'hsapiens')"
                    },
                    "target_namespace": {
                        "type": "string",
                        "description": "Target namespace: ENSG, UNIPROT, ENTREZGENE, etc. (default: ENSG)"
                    }
                },
                "required": ["genes"]
            }
        ),
        Tool(
            name="get_gene_info",
            description="Get gene information including Ensembl ID and description. USE FOR: Gene annotation, name standardization, functional description. ENTITY TYPES: gene. DATA FLOW: Produces standard gene identifiers for downstream analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol or ID"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism code (default: 'hsapiens')"
                    }
                },
                "required": ["gene"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "enrichment_analysis":
        genes = arguments.get("genes", [])
        organism = arguments.get("organism", "hsapiens")
        sources = arguments.get("sources")
        threshold = arguments.get("significance_threshold", 0.05)

        result, error = run_enrichment(genes, organism, sources, threshold)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "convert_gene_ids":
        genes = arguments.get("genes", [])
        organism = arguments.get("organism", "hsapiens")
        target = arguments.get("target_namespace", "ENSG")

        result, error = convert_ids(genes, organism, target)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_gene_info":
        gene = arguments.get("gene", "")
        organism = arguments.get("organism", "hsapiens")

        result, error = get_gene_info(gene, organism)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    logger.info("Starting gProfiler MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
