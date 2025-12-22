"""
NetworkX MCP Server

Provides PPI network analysis using NetworkX library.
Builds networks from STRING-DB data and performs centrality/community analysis.

Tools:
- build_network: Construct network from protein list
- find_hub_proteins: Identify hub proteins
- find_communities: Detect network communities
- calculate_centrality: Compute centrality metrics
- shortest_path: Find shortest path between proteins
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

import requests
import networkx as nx
from community import community_louvain
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("networkx-mcp")

# STRING-DB API
STRING_API = "https://string-db.org/api/json"

# Create MCP server instance
app = Server("networkx")

# Store networks in memory
networks: Dict[str, nx.Graph] = {}
network_counter = 0


def fetch_string_interactions(proteins: List[str], species: int = 9606, score_threshold: int = 400) -> tuple[Optional[List[Dict]], Optional[str]]:
    """
    Fetch protein interactions from STRING-DB.

    Args:
        proteins: List of protein names/IDs
        species: NCBI taxonomy ID (9606 = human)
        score_threshold: Minimum combined score (0-1000)

    Returns:
        (interactions, error_message)
    """
    url = f"{STRING_API}/network"

    params = {
        "identifiers": "%0d".join(proteins),
        "species": species,
        "required_score": score_threshold,
        "caller_identity": "biocoscientist"
    }

    try:
        logger.info(f"Fetching STRING interactions for {len(proteins)} proteins")
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"STRING API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def build_network_from_proteins(proteins: List[str], species: int = 9606, score_threshold: int = 400) -> tuple[Optional[str], Optional[str]]:
    """
    Build network from protein list using STRING-DB.

    Returns:
        (network_id, error_message)
    """
    global network_counter

    interactions, error = fetch_string_interactions(proteins, species, score_threshold)
    if error:
        return None, error

    if not interactions:
        return None, "No interactions found for the given proteins"

    # Create graph
    G = nx.Graph()

    # Add edges from interactions
    for interaction in interactions:
        protein_a = interaction.get("preferredName_A", interaction.get("stringId_A", ""))
        protein_b = interaction.get("preferredName_B", interaction.get("stringId_B", ""))
        score = interaction.get("score", 0)

        if protein_a and protein_b:
            G.add_edge(protein_a, protein_b, weight=score, score=score)

    # Store network
    network_counter += 1
    network_id = f"network_{network_counter}"
    networks[network_id] = G

    return network_id, None


def get_hub_proteins(network_id: str, top_n: int = 10) -> tuple[Optional[Dict], Optional[str]]:
    """
    Find hub proteins by degree centrality.

    Returns:
        (hub_data, error_message)
    """
    if network_id not in networks:
        return None, f"Network not found: {network_id}"

    G = networks[network_id]

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)

    # Sort by centrality
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    hubs = []
    for node, centrality in sorted_nodes[:top_n]:
        hubs.append({
            "protein": node,
            "degree_centrality": round(centrality, 4),
            "degree": G.degree(node),
            "neighbors": list(G.neighbors(node))
        })

    return {
        "network_id": network_id,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "top_hubs": hubs
    }, None


def detect_communities(network_id: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Detect communities using Louvain algorithm.

    Returns:
        (community_data, error_message)
    """
    if network_id not in networks:
        return None, f"Network not found: {network_id}"

    G = networks[network_id]

    if G.number_of_nodes() < 2:
        return None, "Network too small for community detection"

    # Run Louvain community detection
    partition = community_louvain.best_partition(G)

    # Group nodes by community
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    # Calculate modularity
    modularity = community_louvain.modularity(partition, G)

    community_list = []
    for comm_id, members in sorted(communities.items()):
        community_list.append({
            "community_id": comm_id,
            "size": len(members),
            "members": members
        })

    return {
        "network_id": network_id,
        "num_communities": len(communities),
        "modularity": round(modularity, 4),
        "communities": community_list
    }, None


def compute_centrality(network_id: str, metric: str = "degree") -> tuple[Optional[Dict], Optional[str]]:
    """
    Compute centrality metrics.

    Args:
        network_id: Network identifier
        metric: Centrality type (degree, betweenness, closeness, eigenvector)

    Returns:
        (centrality_data, error_message)
    """
    if network_id not in networks:
        return None, f"Network not found: {network_id}"

    G = networks[network_id]

    if metric == "degree":
        centrality = nx.degree_centrality(G)
    elif metric == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif metric == "closeness":
        centrality = nx.closeness_centrality(G)
    elif metric == "eigenvector":
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            centrality = nx.eigenvector_centrality_numpy(G)
    else:
        return None, f"Unknown metric: {metric}. Use: degree, betweenness, closeness, eigenvector"

    # Sort by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    results = [{"protein": node, "centrality": round(val, 4)} for node, val in sorted_nodes]

    return {
        "network_id": network_id,
        "metric": metric,
        "results": results
    }, None


def find_shortest_path(network_id: str, source: str, target: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Find shortest path between two proteins.

    Returns:
        (path_data, error_message)
    """
    if network_id not in networks:
        return None, f"Network not found: {network_id}"

    G = networks[network_id]

    if source not in G:
        return None, f"Source protein not in network: {source}"
    if target not in G:
        return None, f"Target protein not in network: {target}"

    try:
        path = nx.shortest_path(G, source=source, target=target)
        path_length = nx.shortest_path_length(G, source=source, target=target)

        return {
            "network_id": network_id,
            "source": source,
            "target": target,
            "path": path,
            "path_length": path_length,
            "connected": True
        }, None
    except nx.NetworkXNoPath:
        return {
            "network_id": network_id,
            "source": source,
            "target": target,
            "path": [],
            "path_length": -1,
            "connected": False
        }, None


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="build_network",
            description="Build protein interaction network from STRING data for analysis. USE FOR: Network construction, graph analysis preparation. ENTITY TYPES: protein, gene, network. DATA FLOW: Requires protein list, produces NetworkX graph object for hub/community analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "proteins": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of protein/gene names (e.g., ['IL11', 'IL11RA', 'STAT3', 'JAK1'])"
                    },
                    "species": {
                        "type": "integer",
                        "description": "NCBI taxonomy ID (default: 9606 for human)"
                    },
                    "score_threshold": {
                        "type": "integer",
                        "description": "Minimum interaction score 0-1000 (default: 400 = medium confidence)"
                    }
                },
                "required": ["proteins"]
            }
        ),
        Tool(
            name="find_hub_proteins",
            description="Identify hub proteins with high connectivity in the network. USE FOR: Key node identification, drug target prioritization, network centrality. ENTITY TYPES: protein, network. DATA FLOW: Requires network_id from build_network, produces ranked hub proteins.",
            inputSchema={
                "type": "object",
                "properties": {
                    "network_id": {
                        "type": "string",
                        "description": "Network ID from build_network"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top hubs to return (default: 10)"
                    }
                },
                "required": ["network_id"]
            }
        ),
        Tool(
            name="find_communities",
            description="Detect protein communities/modules in the network using Louvain. USE FOR: Module detection, functional clustering, pathway grouping. ENTITY TYPES: protein, pathway, network. DATA FLOW: Requires network_id, produces community assignments with modularity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "network_id": {
                        "type": "string",
                        "description": "Network ID from build_network"
                    }
                },
                "required": ["network_id"]
            }
        ),
        Tool(
            name="calculate_centrality",
            description="Calculate network centrality metrics (degree, betweenness, closeness). USE FOR: Node importance ranking, bottleneck identification. ENTITY TYPES: protein, network. DATA FLOW: Requires network_id, produces centrality scores per node.",
            inputSchema={
                "type": "object",
                "properties": {
                    "network_id": {
                        "type": "string",
                        "description": "Network ID from build_network"
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["degree", "betweenness", "closeness", "eigenvector"],
                        "description": "Centrality metric (default: degree)"
                    }
                },
                "required": ["network_id"]
            }
        ),
        Tool(
            name="shortest_path",
            description="Find shortest path between two proteins in the network. USE FOR: Pathway discovery, signal transduction analysis. ENTITY TYPES: protein, pathway. DATA FLOW: Requires network_id and two protein names, produces path with length.",
            inputSchema={
                "type": "object",
                "properties": {
                    "network_id": {
                        "type": "string",
                        "description": "Network ID from build_network"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source protein name"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target protein name"
                    }
                },
                "required": ["network_id", "source", "target"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "build_network":
        proteins = arguments.get("proteins", [])
        species = arguments.get("species", 9606)
        score_threshold = arguments.get("score_threshold", 400)

        network_id, error = build_network_from_proteins(proteins, species, score_threshold)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]

        G = networks[network_id]
        return [TextContent(type="text", text=json.dumps({
            "network_id": network_id,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "proteins": list(G.nodes())
        }, indent=2))]

    elif name == "find_hub_proteins":
        network_id = arguments.get("network_id", "")
        top_n = arguments.get("top_n", 10)

        result, error = get_hub_proteins(network_id, top_n)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "find_communities":
        network_id = arguments.get("network_id", "")

        result, error = detect_communities(network_id)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "calculate_centrality":
        network_id = arguments.get("network_id", "")
        metric = arguments.get("metric", "degree")

        result, error = compute_centrality(network_id, metric)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "shortest_path":
        network_id = arguments.get("network_id", "")
        source = arguments.get("source", "")
        target = arguments.get("target", "")

        result, error = find_shortest_path(network_id, source, target)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    logger.info("Starting NetworkX MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
