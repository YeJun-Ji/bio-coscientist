"""
IEDB MCP Server

Provides MHC binding prediction using the IEDB API.
Free alternative to NetMHCpan for immunogenicity assessment.

Tools:
- predict_mhc_binding: MHC-I binding prediction
- predict_mhc_ii_binding: MHC-II binding prediction
- scan_protein: Scan protein for epitopes
- list_alleles: Get supported alleles
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from io import StringIO

import requests
import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iedb-mcp")

# IEDB API endpoints
IEDB_MHC_I_API = "http://tools-cluster-interface.iedb.org/tools_api/mhci/"
IEDB_MHC_II_API = "http://tools-cluster-interface.iedb.org/tools_api/mhcii/"

# Common HLA alleles
COMMON_HLA_I = [
    "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*24:02", "HLA-A*11:01",
    "HLA-B*07:02", "HLA-B*08:01", "HLA-B*35:01", "HLA-B*44:02", "HLA-B*51:01",
    "HLA-C*07:01", "HLA-C*07:02", "HLA-C*04:01", "HLA-C*03:04", "HLA-C*06:02"
]

COMMON_HLA_II = [
    "HLA-DRB1*01:01", "HLA-DRB1*03:01", "HLA-DRB1*04:01", "HLA-DRB1*07:01",
    "HLA-DRB1*11:01", "HLA-DRB1*13:01", "HLA-DRB1*15:01",
    "HLA-DPA1*01:03/DPB1*04:01", "HLA-DPA1*02:01/DPB1*01:01",
    "HLA-DQA1*01:02/DQB1*06:02", "HLA-DQA1*05:01/DQB1*02:01"
]

# Create MCP server instance
app = Server("iedb")


def predict_mhc_i(
    sequence: str,
    alleles: List[str] = None,
    method: str = "recommended",
    length: int = 9
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Predict MHC-I binding using IEDB API.

    Args:
        sequence: Peptide or protein sequence
        alleles: List of HLA alleles (default: common alleles)
        method: Prediction method (recommended, netmhcpan_ba, ann, etc.)
        length: Peptide length (8-14)

    Returns:
        (predictions, error_message)
    """
    if alleles is None:
        alleles = ["HLA-A*02:01", "HLA-A*01:01", "HLA-B*07:02"]

    # Format alleles for API
    allele_str = ",".join(alleles)

    data = {
        "method": method,
        "sequence_text": sequence.strip().upper(),
        "allele": allele_str,
        "length": str(length)
    }

    try:
        logger.info(f"Predicting MHC-I binding for sequence of length {len(sequence)}")
        response = requests.post(IEDB_MHC_I_API, data=data, timeout=120)

        if response.status_code == 200:
            # Parse TSV response
            try:
                df = pd.read_csv(StringIO(response.text), sep="\t")

                predictions = []
                for _, row in df.iterrows():
                    pred = {
                        "allele": row.get("allele", ""),
                        "peptide": row.get("peptide", ""),
                        "start": int(row.get("start", 0)),
                        "end": int(row.get("end", 0)),
                        "ic50": float(row.get("ic50", 0)) if pd.notna(row.get("ic50")) else None,
                        "percentile_rank": float(row.get("percentile_rank", 0)) if pd.notna(row.get("percentile_rank")) else None,
                        "score": float(row.get("score", 0)) if pd.notna(row.get("score")) else None
                    }
                    predictions.append(pred)

                # Sort by IC50 (lower is stronger binding)
                predictions.sort(key=lambda x: x["ic50"] if x["ic50"] else float('inf'))

                # Classify binders
                strong_binders = [p for p in predictions if p["ic50"] and p["ic50"] < 50]
                weak_binders = [p for p in predictions if p["ic50"] and 50 <= p["ic50"] < 500]

                return {
                    "sequence_length": len(sequence),
                    "method": method,
                    "alleles": alleles,
                    "total_predictions": len(predictions),
                    "strong_binders": len(strong_binders),
                    "weak_binders": len(weak_binders),
                    "top_epitopes": predictions[:20],  # Top 20 by affinity
                    "interpretation": {
                        "ic50 < 50 nM": "Strong binder (immunogenic)",
                        "50-500 nM": "Weak binder (potentially immunogenic)",
                        "> 500 nM": "Non-binder (likely not immunogenic)"
                    }
                }, None

            except Exception as e:
                return None, f"Failed to parse response: {str(e)}"
        else:
            return None, f"IEDB API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def predict_mhc_ii(
    sequence: str,
    alleles: List[str] = None,
    method: str = "recommended"
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Predict MHC-II binding using IEDB API.

    Returns:
        (predictions, error_message)
    """
    if alleles is None:
        alleles = ["HLA-DRB1*01:01", "HLA-DRB1*03:01", "HLA-DRB1*04:01"]

    allele_str = ",".join(alleles)

    data = {
        "method": method,
        "sequence_text": sequence.strip().upper(),
        "allele": allele_str
    }

    try:
        logger.info(f"Predicting MHC-II binding for sequence of length {len(sequence)}")
        response = requests.post(IEDB_MHC_II_API, data=data, timeout=120)

        if response.status_code == 200:
            try:
                df = pd.read_csv(StringIO(response.text), sep="\t")

                predictions = []
                for _, row in df.iterrows():
                    pred = {
                        "allele": row.get("allele", ""),
                        "peptide": row.get("peptide", ""),
                        "start": int(row.get("start", 0)) if pd.notna(row.get("start")) else 0,
                        "end": int(row.get("end", 0)) if pd.notna(row.get("end")) else 0,
                        "ic50": float(row.get("ic50", 0)) if pd.notna(row.get("ic50")) else None,
                        "percentile_rank": float(row.get("percentile_rank", 0)) if pd.notna(row.get("percentile_rank")) else None
                    }
                    predictions.append(pred)

                predictions.sort(key=lambda x: x["ic50"] if x["ic50"] else float('inf'))

                return {
                    "sequence_length": len(sequence),
                    "method": method,
                    "alleles": alleles,
                    "total_predictions": len(predictions),
                    "top_epitopes": predictions[:20]
                }, None

            except Exception as e:
                return None, f"Failed to parse response: {str(e)}"
        else:
            return None, f"IEDB API error: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def scan_for_epitopes(
    sequence: str,
    alleles: List[str] = None,
    threshold_ic50: float = 500
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Scan protein sequence for potential epitopes.

    Returns:
        (epitope_data, error_message)
    """
    if alleles is None:
        alleles = COMMON_HLA_I[:5]  # Top 5 common alleles

    result, error = predict_mhc_i(sequence, alleles, "recommended", 9)
    if error:
        return None, error

    # Filter for binders
    binders = [
        ep for ep in result.get("top_epitopes", [])
        if ep.get("ic50") and ep["ic50"] < threshold_ic50
    ]

    # Group by position
    hotspots = {}
    for ep in binders:
        pos = ep.get("start", 0)
        if pos not in hotspots:
            hotspots[pos] = {"position": pos, "epitopes": [], "alleles": set()}
        hotspots[pos]["epitopes"].append(ep)
        hotspots[pos]["alleles"].add(ep.get("allele", ""))

    # Convert to list
    hotspot_list = [
        {
            "position": h["position"],
            "num_epitopes": len(h["epitopes"]),
            "num_alleles": len(h["alleles"]),
            "alleles": list(h["alleles"]),
            "best_ic50": min(e.get("ic50", float('inf')) for e in h["epitopes"])
        }
        for h in hotspots.values()
    ]
    hotspot_list.sort(key=lambda x: x["num_alleles"], reverse=True)

    return {
        "sequence_length": len(sequence),
        "total_binders": len(binders),
        "immunogenic_hotspots": hotspot_list[:10],  # Top 10 hotspots
        "risk_assessment": "HIGH" if len(binders) > 20 else "MODERATE" if len(binders) > 5 else "LOW",
        "recommendation": "Consider sequence optimization to remove high-affinity epitopes" if len(binders) > 10 else "Acceptable immunogenicity profile"
    }, None


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="predict_mhc_binding",
            description="Predict MHC-I binding affinity for peptides. USE FOR: Immunogenicity assessment, epitope prediction, binder safety evaluation. ENTITY TYPES: protein, sequence. DATA FLOW: Produces IC50 values for immunogenicity risk assessment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein or peptide sequence"
                    },
                    "alleles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "HLA alleles (e.g., ['HLA-A*02:01', 'HLA-B*07:02']). Default: common alleles"
                    },
                    "method": {
                        "type": "string",
                        "description": "Prediction method: recommended, netmhcpan_ba, ann (default: recommended)"
                    },
                    "length": {
                        "type": "integer",
                        "description": "Peptide length 8-14 (default: 9)"
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="predict_mhc_ii_binding",
            description="Predict MHC-II binding for CD4+ T cell response. USE FOR: Helper T cell epitope prediction, vaccine design. ENTITY TYPES: protein, sequence. DATA FLOW: Produces binding predictions for immune response evaluation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Protein or peptide sequence"
                    },
                    "alleles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "HLA-DR alleles (e.g., ['HLA-DRB1*01:01'])"
                    },
                    "method": {
                        "type": "string",
                        "description": "Prediction method (default: recommended)"
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="scan_protein",
            description="Scan protein for immunogenic epitope hotspots. USE FOR: Therapeutic protein developability, immunogenic region identification. ENTITY TYPES: protein, sequence. DATA FLOW: Produces risk assessment and hotspot positions for sequence optimization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Full protein sequence to scan"
                    },
                    "alleles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "HLA alleles to test (default: 5 common alleles)"
                    },
                    "threshold_ic50": {
                        "type": "number",
                        "description": "IC50 threshold for binder classification (default: 500 nM)"
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="list_alleles",
            description="Get common HLA alleles supported by IEDB. USE FOR: Population coverage analysis, allele selection for prediction. ENTITY TYPES: gene. DATA FLOW: Produces allele list for MHC binding predictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mhc_class": {
                        "type": "string",
                        "enum": ["I", "II"],
                        "description": "MHC class (I or II)"
                    }
                },
                "required": ["mhc_class"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "predict_mhc_binding":
        sequence = arguments.get("sequence", "")
        alleles = arguments.get("alleles")
        method = arguments.get("method", "recommended")
        length = arguments.get("length", 9)

        result, error = predict_mhc_i(sequence, alleles, method, length)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "predict_mhc_ii_binding":
        sequence = arguments.get("sequence", "")
        alleles = arguments.get("alleles")
        method = arguments.get("method", "recommended")

        result, error = predict_mhc_ii(sequence, alleles, method)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "scan_protein":
        sequence = arguments.get("sequence", "")
        alleles = arguments.get("alleles")
        threshold = arguments.get("threshold_ic50", 500)

        result, error = scan_for_epitopes(sequence, alleles, threshold)

        if error:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "list_alleles":
        mhc_class = arguments.get("mhc_class", "I")

        if mhc_class == "I":
            alleles = COMMON_HLA_I
        else:
            alleles = COMMON_HLA_II

        return [TextContent(type="text", text=json.dumps({
            "mhc_class": mhc_class,
            "common_alleles": alleles,
            "count": len(alleles)
        }, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    logger.info("Starting IEDB MCP Server...")

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
