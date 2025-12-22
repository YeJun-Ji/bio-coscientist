"""
Nanopore MCP Server

Provides Oxford Nanopore sequencing data analysis for poly(A) tail and modified base detection.

BAM-based Tools:
- analyze_polya_lengths: Analyze poly(A) tail length distribution from BAM file
- get_alignment_stats: Get BAM file alignment statistics
- detect_modified_bases: Detect modified bases from BAM file with MM/ML tags
- compare_polya_distributions: Compare poly(A) length distributions between samples
- extract_polya_region: Extract poly(A) information for specific genomic region

POD5 Signal Analysis Tools:
- read_pod5_info: Get POD5 file metadata and read statistics
- analyze_polya_signal: Analyze raw signal in poly(A) region for modification signatures
- detect_signal_modifications: Detect modifications from signal patterns
- compare_pod5_signals: Compare signal distributions between two POD5 files
- extract_read_signal: Extract raw signal for specific reads

Note: POD5 processing requires pod5 library. BAM processing requires pysam.
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pod5
    POD5_AVAILABLE = True
except ImportError:
    POD5_AVAILABLE = False

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nanopore-mcp")

# Create MCP server instance
app = Server("nanopore")

# Nanopore signal constants (approximate values for RNA004 chemistry)
# These are reference values - actual values depend on pore chemistry
CANONICAL_A_MEAN = 105.0  # pA - canonical adenosine mean current
MODIFIED_A_MEAN = 90.0    # pA - m6A typically shows lower current
SIGNAL_THRESHOLD_STD = 15.0  # Standard deviation threshold


def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available."""
    return {
        "pysam": PYSAM_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
        "scipy": SCIPY_AVAILABLE,
        "pod5": POD5_AVAILABLE
    }


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available Nanopore analysis tools."""
    return [
        Tool(
            name="analyze_polya_lengths",
            description="Analyze poly(A) tail length distribution from BAM. USE FOR: RNA stability assessment, tail length profiling, regulation analysis. ENTITY TYPES: gene, sequence. DATA FLOW: Produces length distribution statistics for RNA biology study.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bam_file": {
                        "type": "string",
                        "description": "Path to BAM file with poly(A) annotations"
                    },
                    "min_length": {
                        "type": "integer",
                        "description": "Minimum poly(A) length to include (default: 1)",
                        "default": 1
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads to analyze (default: 100000)",
                        "default": 100000
                    }
                },
                "required": ["bam_file"]
            }
        ),
        Tool(
            name="get_alignment_stats",
            description="Get BAM alignment statistics. USE FOR: QC assessment, mapping quality check, coverage analysis. ENTITY TYPES: N/A. DATA FLOW: Produces alignment metrics for sequencing quality evaluation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bam_file": {
                        "type": "string",
                        "description": "Path to BAM file"
                    }
                },
                "required": ["bam_file"]
            }
        ),
        Tool(
            name="detect_modified_bases",
            description="Detect RNA/DNA modifications (5mC, 6mA) from BAM. USE FOR: Epitranscriptomics, methylation analysis, modification profiling. ENTITY TYPES: gene, sequence. DATA FLOW: Produces modification sites for regulatory analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bam_file": {
                        "type": "string",
                        "description": "Path to BAM file with modification tags"
                    },
                    "modification_type": {
                        "type": "string",
                        "description": "Type of modification to detect",
                        "enum": ["5mC", "6mA", "all"],
                        "default": "all"
                    },
                    "min_probability": {
                        "type": "number",
                        "description": "Minimum probability threshold (0-1, default: 0.5)",
                        "default": 0.5
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads to analyze (default: 10000)",
                        "default": 10000
                    }
                },
                "required": ["bam_file"]
            }
        ),
        Tool(
            name="compare_polya_distributions",
            description="Compare poly(A) distributions between samples (KS test). USE FOR: Condition comparison, treatment effect analysis, differential regulation. ENTITY TYPES: gene. DATA FLOW: Produces statistical comparison for biological significance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sample1_bam": {
                        "type": "string",
                        "description": "Path to first sample BAM file"
                    },
                    "sample2_bam": {
                        "type": "string",
                        "description": "Path to second sample BAM file"
                    },
                    "sample1_name": {
                        "type": "string",
                        "description": "Name for first sample (default: 'Sample1')",
                        "default": "Sample1"
                    },
                    "sample2_name": {
                        "type": "string",
                        "description": "Name for second sample (default: 'Sample2')",
                        "default": "Sample2"
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads per sample (default: 50000)",
                        "default": 50000
                    }
                },
                "required": ["sample1_bam", "sample2_bam"]
            }
        ),
        Tool(
            name="extract_polya_region",
            description="Extract poly(A) info for genomic region. USE FOR: Gene-specific tail analysis, locus-level profiling. ENTITY TYPES: gene. DATA FLOW: Produces per-read poly(A) data for targeted analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bam_file": {
                        "type": "string",
                        "description": "Path to BAM file"
                    },
                    "chromosome": {
                        "type": "string",
                        "description": "Chromosome name (e.g., 'chr1', 'NC_045512.2')"
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start position (0-based)"
                    },
                    "end": {
                        "type": "integer",
                        "description": "End position"
                    }
                },
                "required": ["bam_file", "chromosome", "start", "end"]
            }
        ),
        Tool(
            name="analyze_non_a_bases",
            description="Analyze non-A base insertions in poly(A) tails. USE FOR: Tail composition analysis, uridylation detection, mixed tail identification. ENTITY TYPES: gene, sequence. DATA FLOW: Produces base composition for tail modification study.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bam_file": {
                        "type": "string",
                        "description": "Path to BAM file with poly(A) sequence information"
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads to analyze (default: 50000)",
                        "default": 50000
                    }
                },
                "required": ["bam_file"]
            }
        ),
        # ========== POD5 Signal Analysis Tools ==========
        Tool(
            name="read_pod5_info",
            description="Get POD5 file metadata and signal statistics. USE FOR: Raw signal QC, run summary, data overview. ENTITY TYPES: N/A. DATA FLOW: Produces file metadata for signal analysis preparation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod5_file": {
                        "type": "string",
                        "description": "Path to POD5 file"
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads to sample for statistics (default: 100)",
                        "default": 100
                    }
                },
                "required": ["pod5_file"]
            }
        ),
        Tool(
            name="analyze_polya_signal",
            description="Analyze poly(A) raw signal for modification signatures. USE FOR: m6A detection in tails, modification pattern analysis. ENTITY TYPES: gene, sequence. DATA FLOW: Produces signal profiles for modification identification.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod5_file": {
                        "type": "string",
                        "description": "Path to POD5 file"
                    },
                    "tail_length_samples": {
                        "type": "integer",
                        "description": "Number of signal samples to analyze from 3' end (default: 5000, ~1.25 sec at 4kHz)",
                        "default": 5000
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads to analyze (default: 100)",
                        "default": 100
                    }
                },
                "required": ["pod5_file"]
            }
        ),
        Tool(
            name="detect_signal_modifications",
            description="Detect modifications via signal deviation analysis. USE FOR: Base modification calling, current-level profiling. ENTITY TYPES: sequence. DATA FLOW: Produces modification classifications based on z-score thresholds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod5_file": {
                        "type": "string",
                        "description": "Path to POD5 file"
                    },
                    "reference_mean": {
                        "type": "number",
                        "description": "Expected mean signal for canonical A (default: 105.0 pA)",
                        "default": 105.0
                    },
                    "modification_threshold": {
                        "type": "number",
                        "description": "Z-score threshold for modification detection (default: 2.0)",
                        "default": 2.0
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads to analyze (default: 100)",
                        "default": 100
                    }
                },
                "required": ["pod5_file"]
            }
        ),
        Tool(
            name="compare_pod5_signals",
            description="Compare signal distributions between POD5 files. USE FOR: Sample comparison, modification difference detection, treatment effect analysis. ENTITY TYPES: N/A. DATA FLOW: Produces statistical comparison with effect sizes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod5_file1": {
                        "type": "string",
                        "description": "Path to first POD5 file"
                    },
                    "pod5_file2": {
                        "type": "string",
                        "description": "Path to second POD5 file"
                    },
                    "sample1_name": {
                        "type": "string",
                        "description": "Name for first sample (default: 'Sample1')",
                        "default": "Sample1"
                    },
                    "sample2_name": {
                        "type": "string",
                        "description": "Name for second sample (default: 'Sample2')",
                        "default": "Sample2"
                    },
                    "tail_length_samples": {
                        "type": "integer",
                        "description": "Signal samples from 3' end to compare (default: 5000)",
                        "default": 5000
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads per file (default: 100)",
                        "default": 100
                    }
                },
                "required": ["pod5_file1", "pod5_file2"]
            }
        ),
        Tool(
            name="extract_read_signal",
            description="Extract raw signal for specific reads from POD5. USE FOR: Read-level analysis, signal visualization prep. ENTITY TYPES: N/A. DATA FLOW: Produces per-read signal data for detailed investigation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod5_file": {
                        "type": "string",
                        "description": "Path to POD5 file"
                    },
                    "read_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of read IDs to extract (if empty, extracts first N reads)"
                    },
                    "max_reads": {
                        "type": "integer",
                        "description": "Maximum reads to extract if read_ids not specified (default: 5)",
                        "default": 5
                    },
                    "signal_summary_only": {
                        "type": "boolean",
                        "description": "Return only signal statistics, not raw data (default: True)",
                        "default": True
                    }
                },
                "required": ["pod5_file"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        # Check dependencies first
        deps = check_dependencies()
        if not deps["pysam"]:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "pysam library not installed",
                    "install": "pip install pysam"
                })
            )]

        if name == "analyze_polya_lengths":
            result = await analyze_polya_lengths(
                bam_file=arguments["bam_file"],
                min_length=arguments.get("min_length", 1),
                max_reads=arguments.get("max_reads", 100000)
            )
        elif name == "get_alignment_stats":
            result = await get_alignment_stats(
                bam_file=arguments["bam_file"]
            )
        elif name == "detect_modified_bases":
            result = await detect_modified_bases(
                bam_file=arguments["bam_file"],
                modification_type=arguments.get("modification_type", "all"),
                min_probability=arguments.get("min_probability", 0.5),
                max_reads=arguments.get("max_reads", 10000)
            )
        elif name == "compare_polya_distributions":
            if not deps["scipy"]:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "scipy library not installed for statistical tests",
                        "install": "pip install scipy"
                    })
                )]
            result = await compare_polya_distributions(
                sample1_bam=arguments["sample1_bam"],
                sample2_bam=arguments["sample2_bam"],
                sample1_name=arguments.get("sample1_name", "Sample1"),
                sample2_name=arguments.get("sample2_name", "Sample2"),
                max_reads=arguments.get("max_reads", 50000)
            )
        elif name == "extract_polya_region":
            result = await extract_polya_region(
                bam_file=arguments["bam_file"],
                chromosome=arguments["chromosome"],
                start=arguments["start"],
                end=arguments["end"]
            )
        elif name == "analyze_non_a_bases":
            result = await analyze_non_a_bases(
                bam_file=arguments["bam_file"],
                max_reads=arguments.get("max_reads", 50000)
            )
        # ========== POD5 Signal Analysis Tools ==========
        elif name == "read_pod5_info":
            if not POD5_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "pod5 library not installed",
                        "install": "pip install pod5"
                    })
                )]
            result = await read_pod5_info(
                pod5_file=arguments["pod5_file"],
                max_reads=arguments.get("max_reads", 100)
            )
        elif name == "analyze_polya_signal":
            if not POD5_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "pod5 library not installed",
                        "install": "pip install pod5"
                    })
                )]
            result = await analyze_polya_signal(
                pod5_file=arguments["pod5_file"],
                tail_length_samples=arguments.get("tail_length_samples", 5000),
                max_reads=arguments.get("max_reads", 100)
            )
        elif name == "detect_signal_modifications":
            if not POD5_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "pod5 library not installed",
                        "install": "pip install pod5"
                    })
                )]
            result = await detect_signal_modifications(
                pod5_file=arguments["pod5_file"],
                reference_mean=arguments.get("reference_mean", 105.0),
                modification_threshold=arguments.get("modification_threshold", 2.0),
                max_reads=arguments.get("max_reads", 100)
            )
        elif name == "compare_pod5_signals":
            if not POD5_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "pod5 library not installed",
                        "install": "pip install pod5"
                    })
                )]
            if not SCIPY_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "scipy library not installed for statistical tests",
                        "install": "pip install scipy"
                    })
                )]
            result = await compare_pod5_signals(
                pod5_file1=arguments["pod5_file1"],
                pod5_file2=arguments["pod5_file2"],
                sample1_name=arguments.get("sample1_name", "Sample1"),
                sample2_name=arguments.get("sample2_name", "Sample2"),
                tail_length_samples=arguments.get("tail_length_samples", 5000),
                max_reads=arguments.get("max_reads", 100)
            )
        elif name == "extract_read_signal":
            if not POD5_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "pod5 library not installed",
                        "install": "pip install pod5"
                    })
                )]
            result = await extract_read_signal(
                pod5_file=arguments["pod5_file"],
                read_ids=arguments.get("read_ids", []),
                max_reads=arguments.get("max_reads", 5),
                signal_summary_only=arguments.get("signal_summary_only", True)
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def analyze_polya_lengths(
    bam_file: str,
    min_length: int = 1,
    max_reads: int = 100000
) -> Dict[str, Any]:
    """Analyze poly(A) tail length distribution from BAM file."""

    if not os.path.exists(bam_file):
        return {"error": f"BAM file not found: {bam_file}"}

    polya_lengths = []
    read_count = 0
    reads_with_polya = 0

    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read_count >= max_reads:
                    break
                read_count += 1

                # Check for poly(A) tag (common tags: pt, pa, pA)
                polya_len = None

                # Try different poly(A) length tags
                for tag in ["pt", "pa", "pA", "poly_a_length"]:
                    try:
                        polya_len = read.get_tag(tag)
                        break
                    except KeyError:
                        continue

                if polya_len is not None and polya_len >= min_length:
                    polya_lengths.append(polya_len)
                    reads_with_polya += 1

    except Exception as e:
        return {"error": f"Failed to read BAM file: {e}"}

    if not polya_lengths:
        return {
            "bam_file": bam_file,
            "total_reads": read_count,
            "reads_with_polya": 0,
            "warning": "No poly(A) length tags found. File may not have poly(A) annotations."
        }

    # Calculate statistics
    polya_array = np.array(polya_lengths) if NUMPY_AVAILABLE else polya_lengths

    if NUMPY_AVAILABLE:
        stats_dict = {
            "mean": float(np.mean(polya_array)),
            "median": float(np.median(polya_array)),
            "std": float(np.std(polya_array)),
            "min": int(np.min(polya_array)),
            "max": int(np.max(polya_array)),
            "q25": float(np.percentile(polya_array, 25)),
            "q75": float(np.percentile(polya_array, 75))
        }

        # Length distribution bins
        bins = [0, 20, 50, 100, 150, 200, 300, 500, float('inf')]
        hist, _ = np.histogram(polya_array, bins=bins)
        distribution = {
            f"{int(bins[i])}-{int(bins[i+1]) if bins[i+1] != float('inf') else '+'}": int(hist[i])
            for i in range(len(hist))
        }
    else:
        stats_dict = {
            "mean": sum(polya_lengths) / len(polya_lengths),
            "min": min(polya_lengths),
            "max": max(polya_lengths)
        }
        distribution = {}

    return {
        "bam_file": bam_file,
        "total_reads_analyzed": read_count,
        "reads_with_polya": reads_with_polya,
        "polya_fraction": reads_with_polya / read_count if read_count > 0 else 0,
        "statistics": stats_dict,
        "length_distribution": distribution
    }


async def get_alignment_stats(bam_file: str) -> Dict[str, Any]:
    """Get alignment statistics from BAM file."""

    if not os.path.exists(bam_file):
        return {"error": f"BAM file not found: {bam_file}"}

    try:
        stats = {
            "total_reads": 0,
            "mapped_reads": 0,
            "unmapped_reads": 0,
            "primary_alignments": 0,
            "secondary_alignments": 0,
            "supplementary_alignments": 0,
            "read_lengths": [],
            "mapping_qualities": []
        }

        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                stats["total_reads"] += 1

                if read.is_unmapped:
                    stats["unmapped_reads"] += 1
                else:
                    stats["mapped_reads"] += 1

                    if read.is_secondary:
                        stats["secondary_alignments"] += 1
                    elif read.is_supplementary:
                        stats["supplementary_alignments"] += 1
                    else:
                        stats["primary_alignments"] += 1

                    stats["read_lengths"].append(read.query_length or 0)
                    stats["mapping_qualities"].append(read.mapping_quality)

        # Calculate summary statistics
        if stats["read_lengths"] and NUMPY_AVAILABLE:
            lengths = np.array(stats["read_lengths"])
            mquals = np.array(stats["mapping_qualities"])

            result = {
                "bam_file": bam_file,
                "total_reads": stats["total_reads"],
                "mapped_reads": stats["mapped_reads"],
                "unmapped_reads": stats["unmapped_reads"],
                "mapping_rate": stats["mapped_reads"] / stats["total_reads"] if stats["total_reads"] > 0 else 0,
                "primary_alignments": stats["primary_alignments"],
                "secondary_alignments": stats["secondary_alignments"],
                "supplementary_alignments": stats["supplementary_alignments"],
                "read_length_stats": {
                    "mean": float(np.mean(lengths)),
                    "median": float(np.median(lengths)),
                    "min": int(np.min(lengths)),
                    "max": int(np.max(lengths)),
                    "n50": float(np.percentile(lengths, 50))
                },
                "mapping_quality_stats": {
                    "mean": float(np.mean(mquals)),
                    "median": float(np.median(mquals))
                }
            }
        else:
            result = {
                "bam_file": bam_file,
                "total_reads": stats["total_reads"],
                "mapped_reads": stats["mapped_reads"],
                "unmapped_reads": stats["unmapped_reads"],
                "mapping_rate": stats["mapped_reads"] / stats["total_reads"] if stats["total_reads"] > 0 else 0
            }

        # Remove temporary lists
        del stats["read_lengths"]
        del stats["mapping_qualities"]

        return result

    except Exception as e:
        return {"error": f"Failed to read BAM file: {e}"}


async def detect_modified_bases(
    bam_file: str,
    modification_type: str = "all",
    min_probability: float = 0.5,
    max_reads: int = 10000
) -> Dict[str, Any]:
    """Detect modified bases from BAM file with MM/ML tags."""

    if not os.path.exists(bam_file):
        return {"error": f"BAM file not found: {bam_file}"}

    # Modification codes
    mod_codes = {
        "5mC": ["C+m", "m"],  # 5-methylcytosine
        "6mA": ["A+a", "a"],  # N6-methyladenosine
    }

    modifications = defaultdict(lambda: {
        "count": 0,
        "positions": [],
        "probabilities": []
    })

    read_count = 0
    reads_with_mods = 0

    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read_count >= max_reads:
                    break
                read_count += 1

                # Check for modification tags (MM/ML)
                try:
                    mm_tag = read.get_tag("MM")
                    ml_tag = read.get_tag("ML")
                    has_mods = True
                except KeyError:
                    has_mods = False
                    continue

                if has_mods:
                    reads_with_mods += 1

                    # Parse MM tag for modification positions
                    # Format: "C+m,0,1,3;" means methylation at positions 0, 1, 3
                    for mod_spec in mm_tag.split(";"):
                        if not mod_spec:
                            continue

                        parts = mod_spec.split(",")
                        if len(parts) < 2:
                            continue

                        mod_type = parts[0]

                        # Filter by modification type if specified
                        if modification_type != "all":
                            if mod_type not in mod_codes.get(modification_type, []):
                                continue

                        modifications[mod_type]["count"] += len(parts) - 1

    except Exception as e:
        return {"error": f"Failed to read BAM file: {e}"}

    # Format results
    mod_summary = {}
    for mod_type, data in modifications.items():
        mod_summary[mod_type] = {
            "total_modifications": data["count"],
            "avg_per_read": data["count"] / reads_with_mods if reads_with_mods > 0 else 0
        }

    return {
        "bam_file": bam_file,
        "total_reads_analyzed": read_count,
        "reads_with_modifications": reads_with_mods,
        "modification_filter": modification_type,
        "min_probability": min_probability,
        "modifications": mod_summary,
        "note": "Modification detection requires BAM files with MM/ML tags (e.g., from dorado basecaller)"
    }


async def compare_polya_distributions(
    sample1_bam: str,
    sample2_bam: str,
    sample1_name: str = "Sample1",
    sample2_name: str = "Sample2",
    max_reads: int = 50000
) -> Dict[str, Any]:
    """Compare poly(A) length distributions between two samples."""

    # Get poly(A) lengths for both samples
    result1 = await analyze_polya_lengths(sample1_bam, max_reads=max_reads)
    result2 = await analyze_polya_lengths(sample2_bam, max_reads=max_reads)

    if "error" in result1:
        return {"error": f"{sample1_name}: {result1['error']}"}
    if "error" in result2:
        return {"error": f"{sample2_name}: {result2['error']}"}

    # For statistical comparison, we need the raw lengths
    # Re-extract them
    lengths1 = []
    lengths2 = []

    with pysam.AlignmentFile(sample1_bam, "rb") as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i >= max_reads:
                break
            for tag in ["pt", "pa", "pA"]:
                try:
                    lengths1.append(read.get_tag(tag))
                    break
                except KeyError:
                    continue

    with pysam.AlignmentFile(sample2_bam, "rb") as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i >= max_reads:
                break
            for tag in ["pt", "pa", "pA"]:
                try:
                    lengths2.append(read.get_tag(tag))
                    break
                except KeyError:
                    continue

    if not lengths1 or not lengths2:
        return {"error": "Could not extract poly(A) lengths from one or both samples"}

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(lengths1, lengths2)

    # Mann-Whitney U test
    mw_stat, mw_pvalue = stats.mannwhitneyu(lengths1, lengths2, alternative='two-sided')

    return {
        "sample1": {
            "name": sample1_name,
            "bam_file": sample1_bam,
            "n_reads": len(lengths1),
            "mean_polya": float(np.mean(lengths1)),
            "median_polya": float(np.median(lengths1))
        },
        "sample2": {
            "name": sample2_name,
            "bam_file": sample2_bam,
            "n_reads": len(lengths2),
            "mean_polya": float(np.mean(lengths2)),
            "median_polya": float(np.median(lengths2))
        },
        "statistical_tests": {
            "kolmogorov_smirnov": {
                "statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "significant": ks_pvalue < 0.05
            },
            "mann_whitney_u": {
                "statistic": float(mw_stat),
                "p_value": float(mw_pvalue),
                "significant": mw_pvalue < 0.05
            }
        },
        "interpretation": {
            "difference": "significant" if ks_pvalue < 0.05 else "not significant",
            "direction": "Sample1 > Sample2" if np.median(lengths1) > np.median(lengths2) else "Sample2 > Sample1"
        }
    }


async def extract_polya_region(
    bam_file: str,
    chromosome: str,
    start: int,
    end: int
) -> Dict[str, Any]:
    """Extract poly(A) information for a specific genomic region."""

    if not os.path.exists(bam_file):
        return {"error": f"BAM file not found: {bam_file}"}

    polya_data = []

    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(chromosome, start, end):
                polya_len = None
                for tag in ["pt", "pa", "pA"]:
                    try:
                        polya_len = read.get_tag(tag)
                        break
                    except KeyError:
                        continue

                polya_data.append({
                    "read_name": read.query_name,
                    "position": read.reference_start,
                    "strand": "-" if read.is_reverse else "+",
                    "polya_length": polya_len,
                    "mapping_quality": read.mapping_quality
                })

    except Exception as e:
        return {"error": f"Failed to fetch region: {e}"}

    # Calculate statistics for the region
    lengths = [d["polya_length"] for d in polya_data if d["polya_length"] is not None]

    return {
        "bam_file": bam_file,
        "region": f"{chromosome}:{start}-{end}",
        "total_reads": len(polya_data),
        "reads_with_polya": len(lengths),
        "polya_statistics": {
            "mean": float(np.mean(lengths)) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
            "min": int(min(lengths)) if lengths else None,
            "max": int(max(lengths)) if lengths else None
        } if NUMPY_AVAILABLE and lengths else {},
        "reads": polya_data[:100]  # Limit output
    }


async def analyze_non_a_bases(
    bam_file: str,
    max_reads: int = 50000
) -> Dict[str, Any]:
    """Analyze non-A bases within poly(A) tails."""

    if not os.path.exists(bam_file):
        return {"error": f"BAM file not found: {bam_file}"}

    non_a_counts = {"U": 0, "G": 0, "C": 0, "T": 0}
    total_polya_bases = 0
    reads_analyzed = 0

    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if reads_analyzed >= max_reads:
                    break
                reads_analyzed += 1

                # Try to get poly(A) sequence tag if available
                try:
                    polya_seq = read.get_tag("ps")  # poly(A) sequence tag
                except KeyError:
                    continue

                if polya_seq:
                    for base in polya_seq.upper():
                        total_polya_bases += 1
                        if base != "A":
                            non_a_counts[base] = non_a_counts.get(base, 0) + 1

    except Exception as e:
        return {"error": f"Failed to analyze BAM file: {e}"}

    if total_polya_bases == 0:
        return {
            "bam_file": bam_file,
            "reads_analyzed": reads_analyzed,
            "warning": "No poly(A) sequence tags found. Need BAM with 'ps' tag for non-A analysis.",
            "note": "Consider using nanopolish or dorado for poly(A) sequence extraction"
        }

    # Calculate fractions
    non_a_fractions = {
        base: count / total_polya_bases if total_polya_bases > 0 else 0
        for base, count in non_a_counts.items()
    }

    total_non_a = sum(non_a_counts.values())

    return {
        "bam_file": bam_file,
        "reads_analyzed": reads_analyzed,
        "total_polya_bases": total_polya_bases,
        "non_a_bases": {
            "total": total_non_a,
            "fraction": total_non_a / total_polya_bases if total_polya_bases > 0 else 0,
            "by_base": non_a_counts,
            "fractions": non_a_fractions
        },
        "interpretation": {
            "purity": 1 - (total_non_a / total_polya_bases) if total_polya_bases > 0 else 1,
            "most_common_non_a": max(non_a_counts, key=non_a_counts.get) if any(non_a_counts.values()) else None
        }
    }


# ========== POD5 Signal Analysis Functions ==========

async def read_pod5_info(
    pod5_file: str,
    max_reads: int = 100
) -> Dict[str, Any]:
    """Get POD5 file metadata and signal statistics."""

    if not os.path.exists(pod5_file):
        return {"error": f"POD5 file not found: {pod5_file}"}

    try:
        with pod5.Reader(pod5_file) as reader:
            num_reads = reader.num_reads

            # Collect signal statistics from sample of reads
            signal_lengths = []
            signal_means = []
            signal_stds = []
            sample_rate = None

            for i, read in enumerate(reader.reads()):
                if i >= max_reads:
                    break

                signal = read.signal
                signal_lengths.append(len(signal))
                signal_means.append(float(np.mean(signal)))
                signal_stds.append(float(np.std(signal)))

                if sample_rate is None:
                    sample_rate = read.run_info.sample_rate

            return {
                "pod5_file": pod5_file,
                "total_reads": num_reads,
                "reads_sampled": len(signal_lengths),
                "sample_rate_hz": sample_rate,
                "signal_statistics": {
                    "length": {
                        "mean": float(np.mean(signal_lengths)),
                        "min": int(np.min(signal_lengths)),
                        "max": int(np.max(signal_lengths)),
                        "total_samples": int(np.sum(signal_lengths))
                    },
                    "amplitude": {
                        "mean_of_means": float(np.mean(signal_means)),
                        "std_of_means": float(np.std(signal_means)),
                        "mean_of_stds": float(np.mean(signal_stds))
                    }
                },
                "estimated_read_duration_sec": float(np.mean(signal_lengths)) / sample_rate if sample_rate else None
            }

    except Exception as e:
        return {"error": f"Failed to read POD5 file: {e}"}


async def analyze_polya_signal(
    pod5_file: str,
    tail_length_samples: int = 5000,
    max_reads: int = 100
) -> Dict[str, Any]:
    """
    Analyze raw signal in poly(A) region for modification signatures.

    In direct RNA sequencing, the poly(A) tail is at the 3' end of the read.
    Modified bases (m6A, etc.) show different current levels than canonical A.
    """

    if not os.path.exists(pod5_file):
        return {"error": f"POD5 file not found: {pod5_file}"}

    try:
        tail_signals = []
        tail_means = []
        tail_stds = []
        tail_skews = []
        read_ids = []

        with pod5.Reader(pod5_file) as reader:
            for i, read in enumerate(reader.reads()):
                if i >= max_reads:
                    break

                signal = read.signal

                # Extract tail region (last N samples)
                if len(signal) > tail_length_samples:
                    tail_signal = signal[-tail_length_samples:]
                else:
                    tail_signal = signal

                tail_mean = float(np.mean(tail_signal))
                tail_std = float(np.std(tail_signal))

                # Calculate skewness (indicates asymmetry in signal distribution)
                if SCIPY_AVAILABLE and len(tail_signal) > 10:
                    tail_skew = float(scipy_stats.skew(tail_signal))
                else:
                    tail_skew = 0.0

                tail_means.append(tail_mean)
                tail_stds.append(tail_std)
                tail_skews.append(tail_skew)
                read_ids.append(str(read.read_id))

        # Analyze modification signatures
        # Modified A typically shows lower mean and different variance pattern
        mean_of_tail_means = float(np.mean(tail_means))
        std_of_tail_means = float(np.std(tail_means))

        # Classify reads by signal characteristics
        low_signal_reads = sum(1 for m in tail_means if m < MODIFIED_A_MEAN)
        high_signal_reads = sum(1 for m in tail_means if m >= CANONICAL_A_MEAN)
        intermediate_reads = len(tail_means) - low_signal_reads - high_signal_reads

        return {
            "pod5_file": pod5_file,
            "reads_analyzed": len(tail_means),
            "tail_region_samples": tail_length_samples,
            "signal_statistics": {
                "mean": {
                    "overall": mean_of_tail_means,
                    "std": std_of_tail_means,
                    "min": float(np.min(tail_means)),
                    "max": float(np.max(tail_means))
                },
                "variance": {
                    "mean_std": float(np.mean(tail_stds)),
                    "std_of_stds": float(np.std(tail_stds))
                },
                "skewness": {
                    "mean": float(np.mean(tail_skews)),
                    "std": float(np.std(tail_skews))
                }
            },
            "modification_indicators": {
                "low_signal_reads": low_signal_reads,
                "high_signal_reads": high_signal_reads,
                "intermediate_reads": intermediate_reads,
                "potential_modification_fraction": low_signal_reads / len(tail_means) if tail_means else 0,
                "reference_thresholds": {
                    "canonical_A_mean": CANONICAL_A_MEAN,
                    "modified_A_mean": MODIFIED_A_MEAN
                }
            },
            "interpretation": {
                "signal_homogeneity": "homogeneous" if std_of_tail_means < 10 else "heterogeneous",
                "modification_evidence": "strong" if low_signal_reads / len(tail_means) > 0.3 else
                                        "moderate" if low_signal_reads / len(tail_means) > 0.1 else "weak"
            }
        }

    except Exception as e:
        return {"error": f"Failed to analyze POD5 file: {e}"}


async def detect_signal_modifications(
    pod5_file: str,
    reference_mean: float = 105.0,
    modification_threshold: float = 2.0,
    max_reads: int = 100
) -> Dict[str, Any]:
    """
    Detect potential base modifications by analyzing signal level deviations.

    Modified bases show different current levels than canonical bases.
    Uses z-score to identify significant deviations from reference.
    """

    if not os.path.exists(pod5_file):
        return {"error": f"POD5 file not found: {pod5_file}"}

    try:
        # First pass: collect all tail signal means to establish baseline
        all_tail_means = []
        read_data = []

        with pod5.Reader(pod5_file) as reader:
            for i, read in enumerate(reader.reads()):
                if i >= max_reads:
                    break

                signal = read.signal
                # Use last 5000 samples as tail region
                tail_signal = signal[-5000:] if len(signal) > 5000 else signal

                tail_mean = float(np.mean(tail_signal))
                tail_std = float(np.std(tail_signal))

                all_tail_means.append(tail_mean)
                read_data.append({
                    "read_id": str(read.read_id),
                    "signal_length": len(signal),
                    "tail_mean": tail_mean,
                    "tail_std": tail_std
                })

        if not all_tail_means:
            return {"error": "No reads found in POD5 file"}

        # Calculate z-scores relative to reference
        sample_mean = float(np.mean(all_tail_means))
        sample_std = float(np.std(all_tail_means))

        # Classify reads
        modified_reads = []
        canonical_reads = []
        ambiguous_reads = []

        for rd in read_data:
            z_score = (rd["tail_mean"] - reference_mean) / sample_std if sample_std > 0 else 0

            rd["z_score"] = z_score
            rd["classification"] = (
                "potentially_modified" if z_score < -modification_threshold else
                "canonical" if z_score > modification_threshold else
                "ambiguous"
            )

            if rd["classification"] == "potentially_modified":
                modified_reads.append(rd)
            elif rd["classification"] == "canonical":
                canonical_reads.append(rd)
            else:
                ambiguous_reads.append(rd)

        return {
            "pod5_file": pod5_file,
            "reads_analyzed": len(read_data),
            "reference_parameters": {
                "reference_mean": reference_mean,
                "modification_threshold_zscore": modification_threshold
            },
            "sample_statistics": {
                "mean": sample_mean,
                "std": sample_std,
                "deviation_from_reference": sample_mean - reference_mean
            },
            "classification_results": {
                "potentially_modified": len(modified_reads),
                "canonical": len(canonical_reads),
                "ambiguous": len(ambiguous_reads),
                "modification_rate": len(modified_reads) / len(read_data) if read_data else 0
            },
            "top_modified_reads": sorted(modified_reads, key=lambda x: x["z_score"])[:10],
            "interpretation": {
                "overall_modification_level": (
                    "high" if len(modified_reads) / len(read_data) > 0.3 else
                    "moderate" if len(modified_reads) / len(read_data) > 0.1 else
                    "low"
                ),
                "signal_shift": (
                    "lower_than_canonical" if sample_mean < reference_mean else
                    "higher_than_canonical"
                ),
                "note": "Lower signal levels typically indicate m6A or other modifications in poly(A)"
            }
        }

    except Exception as e:
        return {"error": f"Failed to detect modifications: {e}"}


async def compare_pod5_signals(
    pod5_file1: str,
    pod5_file2: str,
    sample1_name: str = "Sample1",
    sample2_name: str = "Sample2",
    tail_length_samples: int = 5000,
    max_reads: int = 100
) -> Dict[str, Any]:
    """
    Compare signal distributions between two POD5 files.

    Useful for comparing CRE vs control, or active vs inactive samples.
    """

    if not os.path.exists(pod5_file1):
        return {"error": f"POD5 file not found: {pod5_file1}"}
    if not os.path.exists(pod5_file2):
        return {"error": f"POD5 file not found: {pod5_file2}"}

    def extract_tail_signals(pod5_path: str, max_n: int, tail_len: int) -> Tuple[List[float], List[float]]:
        means = []
        stds = []
        with pod5.Reader(pod5_path) as reader:
            for i, read in enumerate(reader.reads()):
                if i >= max_n:
                    break
                signal = read.signal
                tail = signal[-tail_len:] if len(signal) > tail_len else signal
                means.append(float(np.mean(tail)))
                stds.append(float(np.std(tail)))
        return means, stds

    try:
        # Extract signals from both files
        means1, stds1 = extract_tail_signals(pod5_file1, max_reads, tail_length_samples)
        means2, stds2 = extract_tail_signals(pod5_file2, max_reads, tail_length_samples)

        if not means1 or not means2:
            return {"error": "Could not extract signals from one or both files"}

        # Statistical tests
        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(means1, means2)

        # Mann-Whitney U test for median difference
        mw_stat, mw_pvalue = scipy_stats.mannwhitneyu(means1, means2, alternative='two-sided')

        # T-test for mean difference
        t_stat, t_pvalue = scipy_stats.ttest_ind(means1, means2)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(means1)**2 + np.std(means2)**2) / 2)
        cohens_d = (np.mean(means1) - np.mean(means2)) / pooled_std if pooled_std > 0 else 0

        return {
            "comparison": f"{sample1_name} vs {sample2_name}",
            "sample1": {
                "name": sample1_name,
                "file": pod5_file1,
                "n_reads": len(means1),
                "mean_signal": float(np.mean(means1)),
                "std_signal": float(np.std(means1)),
                "median_signal": float(np.median(means1))
            },
            "sample2": {
                "name": sample2_name,
                "file": pod5_file2,
                "n_reads": len(means2),
                "mean_signal": float(np.mean(means2)),
                "std_signal": float(np.std(means2)),
                "median_signal": float(np.median(means2))
            },
            "statistical_tests": {
                "kolmogorov_smirnov": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_pvalue),
                    "significant": ks_pvalue < 0.05
                },
                "mann_whitney_u": {
                    "statistic": float(mw_stat),
                    "p_value": float(mw_pvalue),
                    "significant": mw_pvalue < 0.05
                },
                "t_test": {
                    "statistic": float(t_stat),
                    "p_value": float(t_pvalue),
                    "significant": t_pvalue < 0.05
                }
            },
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": (
                    "large" if abs(cohens_d) > 0.8 else
                    "medium" if abs(cohens_d) > 0.5 else
                    "small" if abs(cohens_d) > 0.2 else
                    "negligible"
                ),
                "direction": f"{sample1_name} > {sample2_name}" if cohens_d > 0 else f"{sample2_name} > {sample1_name}"
            },
            "interpretation": {
                "distribution_difference": "significant" if ks_pvalue < 0.05 else "not significant",
                "mean_difference": float(np.mean(means1) - np.mean(means2)),
                "biological_significance": (
                    "Samples show different poly(A) signal profiles, suggesting different modification patterns"
                    if ks_pvalue < 0.05 and abs(cohens_d) > 0.5 else
                    "Samples show similar poly(A) signal profiles"
                )
            }
        }

    except Exception as e:
        return {"error": f"Failed to compare POD5 files: {e}"}


async def extract_read_signal(
    pod5_file: str,
    read_ids: List[str] = None,
    max_reads: int = 5,
    signal_summary_only: bool = True
) -> Dict[str, Any]:
    """
    Extract raw signal data for specific reads from POD5 file.

    If read_ids is empty, extracts first N reads.
    """

    if not os.path.exists(pod5_file):
        return {"error": f"POD5 file not found: {pod5_file}"}

    if read_ids is None:
        read_ids = []

    try:
        results = []

        with pod5.Reader(pod5_file) as reader:
            if read_ids:
                # Extract specific reads by ID
                for read_id_str in read_ids[:max_reads]:
                    for read in reader.reads():
                        if str(read.read_id) == read_id_str:
                            signal = read.signal

                            read_result = {
                                "read_id": str(read.read_id),
                                "signal_length": len(signal),
                                "sample_rate": read.run_info.sample_rate,
                                "duration_sec": len(signal) / read.run_info.sample_rate,
                                "statistics": {
                                    "mean": float(np.mean(signal)),
                                    "std": float(np.std(signal)),
                                    "min": float(np.min(signal)),
                                    "max": float(np.max(signal)),
                                    "median": float(np.median(signal))
                                }
                            }

                            if not signal_summary_only:
                                # Include raw signal (downsampled for large signals)
                                if len(signal) > 10000:
                                    # Downsample to 10000 points
                                    step = len(signal) // 10000
                                    read_result["signal_downsampled"] = signal[::step].tolist()
                                    read_result["downsample_factor"] = step
                                else:
                                    read_result["signal"] = signal.tolist()

                            results.append(read_result)
                            break
            else:
                # Extract first N reads
                for i, read in enumerate(reader.reads()):
                    if i >= max_reads:
                        break

                    signal = read.signal

                    read_result = {
                        "read_id": str(read.read_id),
                        "signal_length": len(signal),
                        "sample_rate": read.run_info.sample_rate,
                        "duration_sec": len(signal) / read.run_info.sample_rate,
                        "statistics": {
                            "mean": float(np.mean(signal)),
                            "std": float(np.std(signal)),
                            "min": float(np.min(signal)),
                            "max": float(np.max(signal)),
                            "median": float(np.median(signal))
                        }
                    }

                    # Tail region analysis
                    tail_signal = signal[-5000:] if len(signal) > 5000 else signal
                    read_result["tail_statistics"] = {
                        "mean": float(np.mean(tail_signal)),
                        "std": float(np.std(tail_signal))
                    }

                    if not signal_summary_only:
                        if len(signal) > 10000:
                            step = len(signal) // 10000
                            read_result["signal_downsampled"] = signal[::step].tolist()
                            read_result["downsample_factor"] = step
                        else:
                            read_result["signal"] = signal.tolist()

                    results.append(read_result)

        return {
            "pod5_file": pod5_file,
            "reads_extracted": len(results),
            "signal_summary_only": signal_summary_only,
            "reads": results
        }

    except Exception as e:
        return {"error": f"Failed to extract signals: {e}"}


async def main():
    """Run the MCP server."""
    logger.info("Starting Nanopore MCP server...")
    deps = check_dependencies()
    logger.info(f"Dependencies: {deps}")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
