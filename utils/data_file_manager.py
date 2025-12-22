"""
Data File Manager - File-based storage for collected data and analysis results

This module eliminates 90%+ information loss by saving full data to files
instead of truncating for LLM prompts.

Architecture:
- Chain 1 (Collection): Save full data to sources.json
- Chain 2 (Analysis): Save full results to results.json
- Answer Generation: Load only analysis results (Chain 1 data not needed)

File Structure:
    logs/<experiment>/data/
        req_1/
            collection/
                sources.json      # Full Chain 1 data (NO truncation)
                metadata.json     # Entity names, timestamps, tool usage
            analysis/
                results.json      # Full Chain 2 results
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataFileManager:
    """
    Manages file-based storage for collected data and analysis results.

    Key Features:
    - Zero data loss: Saves 100% of collected data to files
    - Efficient: LLM sees only metadata, tools access full files
    - Clean separation: Collection data (Chain 1) vs Analysis results (Chain 2)
    """

    def __init__(self, experiment_dir: str):
        """
        Initialize DataFileManager.

        Args:
            experiment_dir: Path to logs/<experiment>_<timestamp>/ directory
        """
        self.experiment_dir = experiment_dir
        self.data_dir = os.path.join(experiment_dir, "data")  # Simplified: /data/req_X instead of /data/requirements/req_X

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        logger.info(f"DataFileManager initialized: {self.data_dir}")

    def save_collection_data(
        self,
        requirement_id: str,
        collected_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save Chain 1 collection data to files WITHOUT truncation.

        Args:
            requirement_id: Requirement ID (e.g., "1", "2")
            collected_data: Full dict from _execute_unified_collection_analysis()
                {
                    "sources": {...},           # MCP tool results
                    "entity_analysis": {...},   # Entity extraction results
                    "tool_usage": {...}         # Tool call tracking
                }

        Returns:
            {
                "sources_file": "data/requirements/req_1/collection/sources.json",
                "metadata_file": "data/requirements/req_1/collection/metadata.json"
            }
        """
        req_dir = os.path.join(self.data_dir, f"req_{requirement_id}")
        collection_dir = os.path.join(req_dir, "collection")
        os.makedirs(collection_dir, exist_ok=True)

        # 1. Save FULL sources.json (NO truncation)
        sources_file = os.path.join(collection_dir, "sources.json")
        try:
            with open(sources_file, 'w', encoding='utf-8') as f:
                json.dump(
                    collected_data.get("sources", {}),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )
            logger.info(f"Saved collection sources: {sources_file}")
        except Exception as e:
            logger.error(f"Failed to save sources.json: {e}")
            raise

        # 2. Save metadata.json (lightweight: entity names, timestamps, tool usage)
        metadata_file = os.path.join(collection_dir, "metadata.json")

        # Extract entity names only (lightweight, ~100 bytes)
        entity_names = []
        entity_analysis = collected_data.get("entity_analysis", {})
        for entity in entity_analysis.get("primary_entities", []):
            entity_names.append({
                "type": entity.get("type", "unknown"),
                "name": entity.get("name", "")
            })

        metadata = {
            "requirement_id": requirement_id,
            "timestamp": datetime.now().isoformat(),
            "tool_usage": collected_data.get("tool_usage", {}),
            "entity_names": entity_names  # Only names, not full entity_analysis
        }

        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved collection metadata: {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata.json: {e}")
            raise

        return {
            "sources_file": sources_file,
            "metadata_file": metadata_file
        }

    def save_analysis_data(
        self,
        requirement_id: str,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save Chain 2 analysis results to files WITHOUT truncation.

        Args:
            requirement_id: Requirement ID
            analysis_results: Full dict from Chain 2 analysis
                {
                    "esmfold_predict": {...},
                    "rosetta_energy": {...},
                    ...
                }

        Returns:
            {
                "results_file": "data/requirements/req_1/analysis/results.json"
            }
        """
        req_dir = os.path.join(self.data_dir, f"req_{requirement_id}")
        analysis_dir = os.path.join(req_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        results_file = os.path.join(analysis_dir, "results.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(
                    analysis_results,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )
            logger.info(f"Saved analysis results: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results.json: {e}")
            raise

        return {"results_file": results_file}

    def load_collection_data(self, requirement_id: str) -> Dict[str, Any]:
        """
        Load full collection data (for tools that need it).

        Args:
            requirement_id: Requirement ID

        Returns:
            Full sources dict from Chain 1, or {} if not found
        """
        sources_file = os.path.join(
            self.data_dir,
            f"req_{requirement_id}",
            "collection",
            "sources.json"
        )

        if not os.path.exists(sources_file):
            logger.warning(f"Collection data not found: {sources_file}")
            return {}

        try:
            with open(sources_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded collection data: {sources_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load collection data: {e}")
            return {}

    def load_metadata(self, requirement_id: str) -> Dict[str, Any]:
        """
        Load metadata (entity names for Chain 2 prompt).

        Args:
            requirement_id: Requirement ID

        Returns:
            Metadata dict with entity_names, or {} if not found
        """
        metadata_file = os.path.join(
            self.data_dir,
            f"req_{requirement_id}",
            "collection",
            "metadata.json"
        )

        if not os.path.exists(metadata_file):
            logger.warning(f"Metadata not found: {metadata_file}")
            return {}

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded metadata: {metadata_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}

    def load_analysis_results(self, requirement_id: str) -> Dict[str, Any]:
        """
        Load full analysis results (for Answer Gen).

        Args:
            requirement_id: Requirement ID

        Returns:
            Full analysis results dict from Chain 2, or {} if not found
        """
        results_file = os.path.join(
            self.data_dir,
            f"req_{requirement_id}",
            "analysis",
            "results.json"
        )

        if not os.path.exists(results_file):
            logger.warning(f"Analysis results not found: {results_file}")
            return {}

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded analysis results: {results_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load analysis results: {e}")
            return {}

    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to file

        Returns:
            File size in bytes, or 0 if file doesn't exist
        """
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return 0

    def save_iteration_result(
        self,
        requirement_id: str,
        iteration: int,
        result_key: str,
        result: Any
    ) -> str:
        """
        Save individual tool result from ReAct iteration to file.

        This prevents data loss when truncating results for LLM conversation.
        Full results are saved to: data/req_X/analysis/iter_N/result_key.json

        Args:
            requirement_id: Requirement ID (e.g., "1", "2")
            iteration: ReAct iteration number (1, 2, 3, ...)
            result_key: Tool result key (e.g., "run_pandas_code_tool_1")
            result: Full tool result (dict or any serializable object)

        Returns:
            File path where result was saved
        """
        req_dir = os.path.join(self.data_dir, f"req_{requirement_id}")
        iteration_dir = os.path.join(req_dir, "analysis", f"iter_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)

        # Clean result key for filename (replace special chars)
        safe_key = result_key.replace("/", "_").replace("\\", "_")
        result_file = os.path.join(iteration_dir, f"{safe_key}.json")

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )
            logger.debug(f"Saved iteration result: {result_file}")
        except Exception as e:
            logger.error(f"Failed to save iteration result: {e}")
            raise

        return result_file
