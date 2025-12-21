"""
Threshold Loader - Domain-agnostic tool threshold validation

This module loads and validates tool metric thresholds from external JSON configuration,
enabling domain-agnostic evaluation without hardcoded thresholds in agent code.

Key Features:
- Load thresholds from config/tool_thresholds.json
- Validate metric interpretations (e.g., "E-value 1e-6 is high confidence")
- Check if values are within valid ranges (e.g., pLDDT must be 0-100)
- Works for ALL domains: protein, pathway, drug_discovery, disease

Usage:
    registry = ThresholdRegistry("config/tool_thresholds.json")

    # Validate interpretation
    is_valid = registry.validate_interpretation(
        tool_name="blast",
        metric_name="e_value",
        value=1e-6,
        claimed_level="high"
    )  # Returns: True (1e-6 <= 1e-5)

    # Check value range
    is_in_range = registry.is_value_in_range(
        tool_name="esmfold",
        metric_name="plddt",
        value=85.3
    )  # Returns: True (85.3 in [0, 100])
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ThresholdRegistry:
    """
    Load and query tool metric thresholds from JSON configuration.

    Enables domain-agnostic evaluation by externalizing all tool-specific
    thresholds to a configuration file.
    """

    def __init__(self, config_path: str):
        """
        Initialize threshold registry from JSON config file.

        Args:
            config_path: Path to tool_thresholds.json

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Threshold config not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in threshold config: {e}")

        self.tools = self.config.get("tools", {})
        self.version = self.config.get("version", "unknown")

        logger.info(f"Loaded threshold registry v{self.version}: {len(self.tools)} tools")

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """
        Get full configuration for a tool.

        Args:
            tool_name: Name of tool (e.g., "blast", "esmfold")

        Returns:
            Tool configuration dict or None if not found
        """
        return self.tools.get(tool_name)

    def get_metric_config(self, tool_name: str, metric_name: str) -> Optional[Dict]:
        """
        Get configuration for a specific metric.

        Args:
            tool_name: Name of tool
            metric_name: Name of metric (e.g., "e_value", "plddt")

        Returns:
            Metric configuration dict or None if not found
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return None

        metrics = tool.get("metrics", {})
        return metrics.get(metric_name)

    def get_metric_threshold(
        self,
        tool_name: str,
        metric_name: str,
        confidence_level: str
    ) -> Optional[Dict]:
        """
        Get threshold for a specific tool/metric/confidence level.

        Args:
            tool_name: Tool name (case-insensitive)
            metric_name: Metric name (case-insensitive)
            confidence_level: Confidence level (e.g., "high", "medium", "low")

        Returns:
            Threshold dict with "operator" and "value"/"min"/"max", or None

        Example:
            >>> registry.get_metric_threshold("blast", "e_value", "high")
            {"operator": "<=", "value": 1e-5, "description": "High confidence homology"}
        """
        # Case-insensitive lookup
        tool_name_lower = tool_name.lower()
        metric_name_lower = metric_name.lower()

        metric = self.get_metric_config(tool_name_lower, metric_name_lower)
        if not metric:
            logger.debug(f"Metric not found: {tool_name}.{metric_name}")
            return None

        interpretations = metric.get("interpretation", {})
        threshold = interpretations.get(confidence_level)

        if not threshold:
            logger.debug(f"Confidence level '{confidence_level}' not found for {tool_name}.{metric_name}")
            return None

        return threshold

    def validate_interpretation(
        self,
        tool_name: str,
        metric_name: str,
        value: float,
        claimed_level: str
    ) -> bool:
        """
        Validate if a claimed confidence level matches the actual metric value.

        This is the core validation method used by LogVerificationAgent.

        Args:
            tool_name: Tool name
            metric_name: Metric name
            value: Actual metric value
            claimed_level: Claimed confidence level (e.g., "high")

        Returns:
            True if interpretation is valid, False otherwise

        Examples:
            >>> # E-value 1e-6 is indeed "high" (1e-6 <= 1e-5)
            >>> validate_interpretation("blast", "e_value", 1e-6, "high")
            True

            >>> # E-value 0.01 is NOT "high" (0.01 > 1e-5)
            >>> validate_interpretation("blast", "e_value", 0.01, "high")
            False

            >>> # pLDDT 85 is "high" (85 >= 70)
            >>> validate_interpretation("esmfold", "plddt", 85, "high")
            True
        """
        threshold = self.get_metric_threshold(
            tool_name, metric_name, claimed_level
        )

        if not threshold:
            # Unknown tool/metric/level - can't validate
            logger.warning(
                f"Cannot validate: unknown threshold for "
                f"{tool_name}.{metric_name}.{claimed_level}"
            )
            return False

        operator = threshold.get("operator")

        try:
            # Comparison operators
            if operator == "<=":
                return value <= threshold["value"]
            elif operator == ">=":
                return value >= threshold["value"]
            elif operator == "<":
                return value < threshold["value"]
            elif operator == ">":
                return value > threshold["value"]
            elif operator == "==":
                return abs(value - threshold["value"]) < 1e-9  # Float comparison
            elif operator == "between":
                return threshold["min"] <= value <= threshold["max"]
            else:
                logger.error(f"Unknown operator: {operator}")
                return False
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error validating interpretation: {e}")
            return False

    def get_valid_range(self, tool_name: str, metric_name: str) -> Optional[Tuple[float, float]]:
        """
        Get valid range for a metric (for pre-check validation).

        Args:
            tool_name: Tool name
            metric_name: Metric name

        Returns:
            Tuple of (min, max) or None if not found

        Example:
            >>> registry.get_valid_range("esmfold", "plddt")
            (0, 100)
        """
        metric = self.get_metric_config(tool_name.lower(), metric_name.lower())
        if not metric:
            return None

        range_spec = metric.get("range")
        if not range_spec or len(range_spec) != 2:
            return None

        return tuple(range_spec)

    def is_value_in_range(
        self,
        tool_name: str,
        metric_name: str,
        value: float
    ) -> bool:
        """
        Check if a value is within the valid range for a metric.

        Used in pre-check phase to reject obviously invalid values.

        Args:
            tool_name: Tool name
            metric_name: Metric name
            value: Value to check

        Returns:
            True if value is in valid range, False otherwise

        Example:
            >>> registry.is_value_in_range("esmfold", "plddt", 85.3)
            True  # 85.3 in [0, 100]

            >>> registry.is_value_in_range("esmfold", "plddt", 150)
            False  # 150 > 100
        """
        valid_range = self.get_valid_range(tool_name, metric_name)
        if not valid_range:
            # No range defined - can't validate
            logger.debug(f"No range defined for {tool_name}.{metric_name}, skipping check")
            return True  # Assume valid if no range specified

        min_val, max_val = valid_range
        return min_val <= value <= max_val

    def get_all_metrics_for_tool(self, tool_name: str) -> List[str]:
        """
        Get list of all metrics defined for a tool.

        Args:
            tool_name: Tool name

        Returns:
            List of metric names
        """
        tool = self.tools.get(tool_name.lower())
        if not tool:
            return []

        metrics = tool.get("metrics", {})
        return list(metrics.keys())

    def get_supported_tools(self) -> List[str]:
        """
        Get list of all supported tools.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tools_by_domain(self, domain: str) -> List[str]:
        """
        Get all tools for a specific domain.

        Args:
            domain: Domain name (e.g., "protein", "pathway", "drug_discovery", "disease")

        Returns:
            List of tool names in that domain
        """
        return [
            tool_name
            for tool_name, tool_config in self.tools.items()
            if tool_config.get("domain") == domain
        ]

    def suggest_confidence_level(
        self,
        tool_name: str,
        metric_name: str,
        value: float
    ) -> Optional[str]:
        """
        Suggest appropriate confidence level for a given metric value.

        Useful for auto-generating interpretations or detecting over-interpretation.

        Args:
            tool_name: Tool name
            metric_name: Metric name
            value: Metric value

        Returns:
            Suggested confidence level (e.g., "high", "medium", "low") or None

        Example:
            >>> registry.suggest_confidence_level("blast", "e_value", 1e-6)
            "high"  # 1e-6 <= 1e-5

            >>> registry.suggest_confidence_level("blast", "e_value", 0.01)
            "low"  # 0.01 > 0.001
        """
        metric = self.get_metric_config(tool_name.lower(), metric_name.lower())
        if not metric:
            return None

        interpretations = metric.get("interpretation", {})

        # Check each level in priority order (usually high -> medium -> low)
        # But need to handle different operator types
        for level, threshold in interpretations.items():
            if self._check_threshold(value, threshold):
                return level

        return None

    def _check_threshold(self, value: float, threshold: Dict) -> bool:
        """
        Internal helper to check if value meets threshold criteria.

        Args:
            value: Value to check
            threshold: Threshold configuration dict

        Returns:
            True if value meets threshold
        """
        operator = threshold.get("operator")

        try:
            if operator == "<=":
                return value <= threshold["value"]
            elif operator == ">=":
                return value >= threshold["value"]
            elif operator == "<":
                return value < threshold["value"]
            elif operator == ">":
                return value > threshold["value"]
            elif operator == "==":
                return abs(value - threshold["value"]) < 1e-9
            elif operator == "between":
                return threshold["min"] <= value <= threshold["max"]
            else:
                return False
        except (KeyError, TypeError):
            return False

    def get_metric_unit(self, tool_name: str, metric_name: str) -> Optional[str]:
        """
        Get unit for a metric (e.g., "kcal/mol", "Angstrom", "percent").

        Args:
            tool_name: Tool name
            metric_name: Metric name

        Returns:
            Unit string or None
        """
        metric = self.get_metric_config(tool_name.lower(), metric_name.lower())
        if not metric:
            return None

        return metric.get("unit")

    def get_metric_description(self, tool_name: str, metric_name: str) -> Optional[str]:
        """
        Get description for a metric.

        Args:
            tool_name: Tool name
            metric_name: Metric name

        Returns:
            Description string or None
        """
        metric = self.get_metric_config(tool_name.lower(), metric_name.lower())
        if not metric:
            return None

        return metric.get("description")
