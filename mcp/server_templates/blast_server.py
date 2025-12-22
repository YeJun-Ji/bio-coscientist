"""
BLAST MCP Server

Provides protein/nucleotide sequence similarity search using NCBI BLAST API.
Useful for finding homologous proteins, off-target prediction, and evolutionary analysis.

Tools:
- blastp_search: Submit protein vs protein BLAST search
- check_blast_status: Check job status using RID
- get_blast_results: Retrieve results when ready
- find_similar_proteins: High-level tool that submits, waits, and returns results

API Limitations:
- Max 1 request per 10 seconds
- Max 1 status check per minute per RID
- Max 100 searches per 24 hours (may be throttled above this)
"""

import asyncio
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blast-mcp")

# NCBI BLAST API
BLAST_API = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 10  # seconds between requests

# Create MCP server instance
app = Server("blast")


def rate_limit():
    """Enforce rate limiting for NCBI API"""
    global last_request_time
    current_time = time.time()
    elapsed = current_time - last_request_time

    if elapsed < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - elapsed
        logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)

    last_request_time = time.time()


def validate_sequence(sequence: str) -> tuple:
    """Validate protein sequence format."""
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")

    if len(sequence) == 0:
        return False, "Empty sequence provided"

    if len(sequence) < 10:
        return False, f"Sequence too short: {len(sequence)} residues (min 10)"

    # Check for valid amino acid characters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX*-")
    invalid_chars = set(sequence) - valid_aa

    if invalid_chars:
        return False, f"Invalid characters: {invalid_chars}"

    return True, ""


def parse_rid(response_text: str) -> Optional[str]:
    """Parse RID (Request ID) from BLAST submission response."""
    # Look for RID in response
    rid_match = re.search(r'RID = ([A-Z0-9]+)', response_text)
    if rid_match:
        return rid_match.group(1)

    # Alternative format
    rid_match = re.search(r'QBlastInfoBegin\s+RID\s*=\s*([A-Z0-9]+)', response_text)
    if rid_match:
        return rid_match.group(1)

    return None


def parse_status(response_text: str) -> str:
    """Parse job status from BLAST status check response."""
    if "Status=WAITING" in response_text:
        return "WAITING"
    elif "Status=READY" in response_text:
        return "READY"
    elif "Status=UNKNOWN" in response_text:
        return "UNKNOWN"
    elif "Status=FAILED" in response_text:
        return "FAILED"
    else:
        return "UNKNOWN"


def parse_blast_results(response_text: str) -> Dict[str, Any]:
    """Parse BLAST results from various formats."""
    try:
        # Try JSON format first
        if response_text.strip().startswith('{'):
            return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Parse text format (simplified)
    results = {
        "format": "text",
        "hits": [],
        "raw_output": response_text[:5000]  # Truncate for large outputs
    }

    # Extract basic hit information from text format
    hit_pattern = re.compile(r'>([^\n]+)\n.*?Score\s*=\s*([\d.]+).*?Expect\s*=\s*([^\s,]+)', re.DOTALL)

    for match in hit_pattern.finditer(response_text):
        results["hits"].append({
            "title": match.group(1).strip()[:200],
            "score": float(match.group(2)),
            "evalue": match.group(3)
        })

    results["hit_count"] = len(results["hits"])
    return results


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available BLAST tools."""
    return [
        Tool(
            name="blastp_search",
            description="Submit protein BLAST search against databases. USE FOR: Homolog discovery, off-target prediction, evolutionary analysis. ENTITY TYPES: protein, sequence. DATA FLOW: Produces RID for async result retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein sequence in single-letter amino acid code"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database to search (swissprot, nr, pdb, refseq_protein)",
                        "default": "swissprot",
                        "enum": ["swissprot", "nr", "pdb", "refseq_protein"]
                    },
                    "expect": {
                        "type": "number",
                        "description": "E-value threshold (default: 10)",
                        "default": 10
                    },
                    "max_hits": {
                        "type": "integer",
                        "description": "Maximum number of hits to return (default: 50)",
                        "default": 50
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="check_blast_status",
            description="Check BLAST job status using RID. USE FOR: Job monitoring, async workflow management. ENTITY TYPES: N/A. DATA FLOW: Produces status (WAITING/READY/FAILED) for result retrieval timing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rid": {
                        "type": "string",
                        "description": "Request ID from blastp_search"
                    }
                },
                "required": ["rid"]
            }
        ),
        Tool(
            name="get_blast_results",
            description="Retrieve completed BLAST search results. USE FOR: Homolog retrieval, similarity analysis. ENTITY TYPES: protein, sequence. DATA FLOW: Requires READY RID, produces ranked hit list with E-values.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rid": {
                        "type": "string",
                        "description": "Request ID from blastp_search"
                    },
                    "format_type": {
                        "type": "string",
                        "description": "Output format (JSON2, XML2, Text)",
                        "default": "Text",
                        "enum": ["JSON2", "XML2", "Text"]
                    }
                },
                "required": ["rid"]
            }
        ),
        Tool(
            name="find_similar_proteins",
            description="Find similar proteins (all-in-one BLAST). USE FOR: Off-target assessment, homolog discovery, selectivity analysis. ENTITY TYPES: protein, sequence. DATA FLOW: Produces ranked similar proteins for specificity evaluation. Takes 1-5 min.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein sequence in single-letter amino acid code"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database to search",
                        "default": "swissprot",
                        "enum": ["swissprot", "pdb"]
                    },
                    "max_hits": {
                        "type": "integer",
                        "description": "Maximum number of hits to return",
                        "default": 10
                    },
                    "timeout_minutes": {
                        "type": "integer",
                        "description": "Maximum wait time in minutes (default: 5)",
                        "default": 5
                    }
                },
                "required": ["sequence"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "blastp_search":
        return await blastp_search(
            arguments.get("sequence", ""),
            arguments.get("database", "swissprot"),
            arguments.get("expect", 10),
            arguments.get("max_hits", 50)
        )
    elif name == "check_blast_status":
        return await check_blast_status(arguments.get("rid", ""))
    elif name == "get_blast_results":
        return await get_blast_results(
            arguments.get("rid", ""),
            arguments.get("format_type", "Text")
        )
    elif name == "find_similar_proteins":
        return await find_similar_proteins(
            arguments.get("sequence", ""),
            arguments.get("database", "swissprot"),
            arguments.get("max_hits", 10),
            arguments.get("timeout_minutes", 5)
        )
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def blastp_search(
    sequence: str,
    database: str = "swissprot",
    expect: float = 10,
    max_hits: int = 50
) -> List[TextContent]:
    """Submit a BLAST search and return the RID."""

    # Clean and validate sequence
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")
    is_valid, error = validate_sequence(sequence)

    if not is_valid:
        return [TextContent(type="text", text=json.dumps({"error": error}))]

    logger.info(f"Submitting BLAST search: {len(sequence)} residues against {database}")

    # Rate limit
    rate_limit()

    # Submit search
    params = {
        "CMD": "Put",
        "PROGRAM": "blastp",
        "DATABASE": database,
        "QUERY": sequence,
        "EXPECT": str(expect),
        "HITLIST_SIZE": str(max_hits),
        "FORMAT_TYPE": "Text",
        "EMAIL": "biocoscientist@research.ai",
        "TOOL": "biocoscientist-mcp"
    }

    try:
        response = requests.get(BLAST_API, params=params, timeout=30)
        response.raise_for_status()

        rid = parse_rid(response.text)

        if rid:
            logger.info(f"BLAST search submitted: RID={rid}")
            return [TextContent(type="text", text=json.dumps({
                "status": "submitted",
                "rid": rid,
                "database": database,
                "sequence_length": len(sequence),
                "message": "Use check_blast_status to monitor progress, then get_blast_results when ready"
            }))]
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": "Failed to get RID from BLAST response",
                "response_preview": response.text[:500]
            }))]

    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"BLAST API request failed: {str(e)}"
        }))]


async def check_blast_status(rid: str) -> List[TextContent]:
    """Check the status of a BLAST search."""

    if not rid:
        return [TextContent(type="text", text=json.dumps({"error": "RID is required"}))]

    logger.info(f"Checking BLAST status for RID={rid}")

    # Rate limit
    rate_limit()

    params = {
        "CMD": "Get",
        "RID": rid,
        "FORMAT_OBJECT": "SearchInfo"
    }

    try:
        response = requests.get(BLAST_API, params=params, timeout=30)
        response.raise_for_status()

        status = parse_status(response.text)

        result = {
            "rid": rid,
            "status": status
        }

        if status == "READY":
            result["message"] = "Results are ready. Use get_blast_results to retrieve them."
        elif status == "WAITING":
            result["message"] = "Search is still running. Check again in 1 minute."
        elif status == "FAILED":
            result["message"] = "Search failed. Please try again."
        else:
            result["message"] = "Unknown status. RID may have expired (24 hour limit)."

        return [TextContent(type="text", text=json.dumps(result))]

    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Status check failed: {str(e)}"
        }))]


async def get_blast_results(rid: str, format_type: str = "Text") -> List[TextContent]:
    """Retrieve BLAST results."""

    if not rid:
        return [TextContent(type="text", text=json.dumps({"error": "RID is required"}))]

    logger.info(f"Retrieving BLAST results for RID={rid}")

    # Rate limit
    rate_limit()

    params = {
        "CMD": "Get",
        "RID": rid,
        "FORMAT_TYPE": format_type
    }

    try:
        response = requests.get(BLAST_API, params=params, timeout=60)
        response.raise_for_status()

        # Parse results
        results = parse_blast_results(response.text)
        results["rid"] = rid
        results["format_requested"] = format_type

        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to retrieve results: {str(e)}"
        }))]


async def find_similar_proteins(
    sequence: str,
    database: str = "swissprot",
    max_hits: int = 10,
    timeout_minutes: int = 5
) -> List[TextContent]:
    """High-level tool: Submit, wait, and return results."""

    # Submit search
    submit_result = await blastp_search(sequence, database, 10, max_hits)
    submit_data = json.loads(submit_result[0].text)

    if "error" in submit_data:
        return submit_result

    rid = submit_data.get("rid")
    if not rid:
        return [TextContent(type="text", text=json.dumps({
            "error": "Failed to get RID from submission"
        }))]

    logger.info(f"Waiting for BLAST results: RID={rid}")

    # Poll for results
    max_checks = timeout_minutes * 6  # Check every 10 seconds

    for i in range(max_checks):
        # Wait before checking (first wait is shorter)
        await asyncio.sleep(10 if i == 0 else 60)

        # Check status
        status_result = await check_blast_status(rid)
        status_data = json.loads(status_result[0].text)

        status = status_data.get("status")
        logger.info(f"BLAST status check {i+1}/{max_checks}: {status}")

        if status == "READY":
            # Get results
            results = await get_blast_results(rid, "Text")
            return results
        elif status == "FAILED":
            return [TextContent(type="text", text=json.dumps({
                "error": "BLAST search failed",
                "rid": rid
            }))]
        elif status == "UNKNOWN":
            return [TextContent(type="text", text=json.dumps({
                "error": "BLAST RID not found or expired",
                "rid": rid
            }))]
        # Otherwise continue waiting

    # Timeout
    return [TextContent(type="text", text=json.dumps({
        "error": f"BLAST search timed out after {timeout_minutes} minutes",
        "rid": rid,
        "message": "You can still check status and retrieve results later using the RID"
    }))]


async def main():
    """Run the MCP server."""
    logger.info("Starting BLAST MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
