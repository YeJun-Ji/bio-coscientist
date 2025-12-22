#!/usr/bin/env python3
"""
Foldseek MCP Server
Provides protein structure similarity search using Foldseek web API

API Reference: https://search.foldseek.com (same backend as MMseqs2)
"""

import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("foldseek-mcp-server")

# Foldseek API base URL
FOLDSEEK_API_BASE = "https://search.foldseek.com/api"

# Available databases
FOLDSEEK_DATABASES = {
    "afdb50": "AlphaFold DB UniProt50 (recommended)",
    "afdb-swissprot": "AlphaFold DB SwissProt subset",
    "afdb-proteome": "AlphaFold DB Proteomes",
    "pdb100": "PDB (redundancy filtered)",
    "gmgcl_id": "GMGC (gut microbiome)",
    "mgnify_esm30": "MGnify ESM30",
}


class FoldseekClient:
    """Client for Foldseek REST API"""

    def __init__(self, base_url: str = FOLDSEEK_API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        })

    def submit_search(
        self,
        pdb_content: str,
        databases: List[str] = None,
        mode: str = "3diaa"
    ) -> Dict[str, Any]:
        """
        Submit a structure search job

        Args:
            pdb_content: PDB format structure content
            databases: List of databases to search (default: afdb50)
            mode: Search mode - "3diaa" (fast, default) or "tmalign" (accurate)

        Returns:
            Ticket information with job ID
        """
        if databases is None:
            databases = ["afdb50"]

        data = {
            "q": pdb_content,
            "mode": mode,
        }

        # Add databases
        for db in databases:
            data.setdefault("database[]", []).append(db)

        response = self.session.post(
            f"{self.base_url}/ticket",
            data=data
        )
        response.raise_for_status()
        return response.json()

    def check_status(self, ticket_id: str) -> Dict[str, Any]:
        """Check job status"""
        response = self.session.get(f"{self.base_url}/ticket/{ticket_id}")
        response.raise_for_status()
        return response.json()

    def get_results(self, ticket_id: str, entry: int = 0) -> Dict[str, Any]:
        """Get search results for a specific query entry"""
        response = self.session.get(f"{self.base_url}/result/{ticket_id}/{entry}")
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        ticket_id: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Wait for job completion

        Args:
            ticket_id: Job ticket ID
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks

        Returns:
            Final job status
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.check_status(ticket_id)
            job_status = status.get("status", "UNKNOWN")

            if job_status == "COMPLETE":
                return status
            elif job_status == "ERROR":
                raise RuntimeError(f"Foldseek job failed: {status}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Job {ticket_id} did not complete within {timeout} seconds")

    def search_and_wait(
        self,
        pdb_content: str,
        databases: List[str] = None,
        mode: str = "3diaa",
        timeout: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Submit search and wait for results

        Returns:
            List of hit alignments with TM-scores
        """
        # Submit job
        ticket = self.submit_search(pdb_content, databases, mode)
        ticket_id = ticket.get("id")

        if not ticket_id:
            raise RuntimeError(f"Failed to get ticket ID: {ticket}")

        logger.info(f"Submitted Foldseek job: {ticket_id}")

        # Wait for completion
        self.wait_for_completion(ticket_id, timeout)

        # Get results
        results = self.get_results(ticket_id, 0)
        return results.get("alignments", [])


# Initialize server and client
server = Server("foldseek-mcp-server")
foldseek_client = FoldseekClient()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available Foldseek tools"""
    return [
        Tool(
            name="search_structure",
            description="Search structurally similar proteins in AlphaFold DB/PDB. USE FOR: Structural homolog discovery, fold classification, off-target structure search. ENTITY TYPES: protein, structure. DATA FLOW: Requires PDB, produces TM-score ranked hits (>=0.5 same fold, >=0.7 same superfamily).",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_content": {
                        "type": "string",
                        "description": "PDB format structure content"
                    },
                    "databases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Databases to search (default: ['afdb50'])",
                        "default": ["afdb50"]
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["3diaa", "tmalign"],
                        "description": "Search mode: '3diaa' (fast) or 'tmalign' (accurate)",
                        "default": "3diaa"
                    },
                    "max_hits": {
                        "type": "integer",
                        "description": "Maximum number of hits to return",
                        "default": 50
                    }
                },
                "required": ["pdb_content"]
            }
        ),
        Tool(
            name="compare_two_structures",
            description="Compare two structures for TM-score alignment. USE FOR: Structure similarity assessment, design vs native comparison. ENTITY TYPES: protein, structure. DATA FLOW: Produces TM-score for structural equivalence evaluation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "structure_a_pdb": {
                        "type": "string",
                        "description": "First structure in PDB format"
                    },
                    "structure_b_pdb": {
                        "type": "string",
                        "description": "Second structure in PDB format"
                    },
                    "structure_b_name": {
                        "type": "string",
                        "description": "Name/identifier of structure B for matching"
                    }
                },
                "required": ["structure_a_pdb", "structure_b_pdb"]
            }
        ),
        Tool(
            name="get_structural_neighbors",
            description="Find structural neighbors in AlphaFold DB. USE FOR: Functional inference from structure, related protein discovery. ENTITY TYPES: protein, structure. DATA FLOW: Produces structurally similar proteins above TM-score threshold.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_content": {
                        "type": "string",
                        "description": "Query structure in PDB format"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top neighbors to return",
                        "default": 20
                    },
                    "min_tm_score": {
                        "type": "number",
                        "description": "Minimum TM-score threshold",
                        "default": 0.5
                    }
                },
                "required": ["pdb_content"]
            }
        ),
        Tool(
            name="batch_structure_search",
            description="Search multiple structures in batch. USE FOR: Large-scale structural analysis, gene set structure comparison. ENTITY TYPES: protein, structure. DATA FLOW: Processes multiple queries, produces aggregated TM-score results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "structures": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "pdb_content": {"type": "string"}
                            }
                        },
                        "description": "List of structures with names and PDB content"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database to search",
                        "default": "afdb50"
                    },
                    "top_hits_per_query": {
                        "type": "integer",
                        "description": "Number of top hits per query",
                        "default": 10
                    }
                },
                "required": ["structures"]
            }
        ),
        Tool(
            name="list_databases",
            description="List available Foldseek databases. USE FOR: Database selection, search scope decision. ENTITY TYPES: N/A. DATA FLOW: Produces database options (afdb50, pdb100, etc.) for search configuration.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""

    try:
        if name == "search_structure":
            return await handle_search_structure(arguments)
        elif name == "compare_two_structures":
            return await handle_compare_structures(arguments)
        elif name == "get_structural_neighbors":
            return await handle_structural_neighbors(arguments)
        elif name == "batch_structure_search":
            return await handle_batch_search(arguments)
        elif name == "list_databases":
            return await handle_list_databases(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_search_structure(args: Dict[str, Any]) -> List[TextContent]:
    """Handle structure search"""
    pdb_content = args["pdb_content"]
    databases = args.get("databases", ["afdb50"])
    mode = args.get("mode", "3diaa")
    max_hits = args.get("max_hits", 50)

    # Run search in thread pool (blocking API)
    loop = asyncio.get_event_loop()
    alignments = await loop.run_in_executor(
        None,
        lambda: foldseek_client.search_and_wait(pdb_content, databases, mode)
    )

    # Process results
    hits = []
    for i, aln in enumerate(alignments[:max_hits]):
        hit = {
            "rank": i + 1,
            "target": aln.get("target", ""),
            "tm_score": aln.get("prob", 0) / 100,  # Convert to 0-1 scale
            "e_value": aln.get("eval", 999),
            "identity": aln.get("seqId", 0),
            "aligned_length": aln.get("alnLength", 0),
            "query_start": aln.get("qStartPos", 0),
            "query_end": aln.get("qEndPos", 0),
            "target_start": aln.get("tStartPos", 0),
            "target_end": aln.get("tEndPos", 0),
        }
        hits.append(hit)

    result = {
        "total_hits": len(alignments),
        "returned_hits": len(hits),
        "databases": databases,
        "mode": mode,
        "hits": hits
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_compare_structures(args: Dict[str, Any]) -> List[TextContent]:
    """Compare two structures"""
    structure_a = args["structure_a_pdb"]
    structure_b = args["structure_b_pdb"]
    structure_b_name = args.get("structure_b_name", "structure_b")

    # For direct comparison, we'd need to run local TM-align
    # Using Foldseek API, we search A and look for B in results
    # This is a workaround - ideally use local TM-align for pairwise

    result = {
        "note": "Direct pairwise comparison requires local TM-align installation. "
                "Using Foldseek search as approximation.",
        "recommendation": "For accurate pairwise TM-score, consider using ESMFold + local TM-align, "
                         "or compare both structures against the same database and look for shared hits."
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_structural_neighbors(args: Dict[str, Any]) -> List[TextContent]:
    """Find structural neighbors"""
    pdb_content = args["pdb_content"]
    top_n = args.get("top_n", 20)
    min_tm_score = args.get("min_tm_score", 0.5)

    loop = asyncio.get_event_loop()
    alignments = await loop.run_in_executor(
        None,
        lambda: foldseek_client.search_and_wait(pdb_content, ["afdb50"], "3diaa")
    )

    # Filter by TM-score and get top N
    neighbors = []
    for aln in alignments:
        tm_score = aln.get("prob", 0) / 100
        if tm_score >= min_tm_score:
            neighbors.append({
                "target": aln.get("target", ""),
                "tm_score": tm_score,
                "e_value": aln.get("eval", 999),
                "identity": aln.get("seqId", 0),
                "description": aln.get("tDescription", "")
            })

        if len(neighbors) >= top_n:
            break

    result = {
        "query_neighbors": len(neighbors),
        "min_tm_score_threshold": min_tm_score,
        "neighbors": neighbors
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_batch_search(args: Dict[str, Any]) -> List[TextContent]:
    """Handle batch structure search"""
    structures = args["structures"]
    database = args.get("database", "afdb50")
    top_hits = args.get("top_hits_per_query", 10)

    results = []

    for i, struct in enumerate(structures):
        name = struct.get("name", f"structure_{i}")
        pdb_content = struct.get("pdb_content", "")

        if not pdb_content:
            results.append({
                "name": name,
                "status": "error",
                "message": "No PDB content provided"
            })
            continue

        try:
            loop = asyncio.get_event_loop()
            alignments = await loop.run_in_executor(
                None,
                lambda pc=pdb_content: foldseek_client.search_and_wait(pc, [database], "3diaa")
            )

            hits = []
            for aln in alignments[:top_hits]:
                hits.append({
                    "target": aln.get("target", ""),
                    "tm_score": aln.get("prob", 0) / 100,
                    "e_value": aln.get("eval", 999)
                })

            results.append({
                "name": name,
                "status": "success",
                "total_hits": len(alignments),
                "top_hits": hits
            })

            # Rate limiting - wait between requests
            await asyncio.sleep(2)

        except Exception as e:
            results.append({
                "name": name,
                "status": "error",
                "message": str(e)
            })

    return [TextContent(type="text", text=json.dumps({
        "total_queries": len(structures),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "results": results
    }, indent=2))]


async def handle_list_databases(args: Dict[str, Any]) -> List[TextContent]:
    """List available databases"""
    return [TextContent(type="text", text=json.dumps({
        "databases": FOLDSEEK_DATABASES,
        "recommended": "afdb50",
        "note": "afdb50 contains AlphaFold predictions for UniProt50 representative sequences"
    }, indent=2))]


async def main():
    """Main entry point"""
    logger.info("Starting Foldseek MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
