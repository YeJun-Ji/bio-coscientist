"""
Automatic Parameter Mapper - Schema-based & Pattern-based Parameter Resolution

This module automatically maps collected data to tool parameters without
requiring individual tool configurations. It uses:

1. MCP Server's inputSchema (already provided by servers)
2. Parameter name patterns (semantic matching)
3. Data type inference
4. Context from collected_data and input_files

Architecture:
- No manual mapping per tool required
- Pattern-based rules cover most cases
- Schema validation ensures correctness
- Fallback to LLM-suggested values when needed
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParameterPattern:
    """Pattern rule for automatic parameter mapping"""
    pattern: str  # Regex pattern for parameter name
    data_source: str  # Where to find data: "input_files", "collected_data", "entity_analysis"
    extraction_path: str  # JSONPath-like extraction pattern
    transform: Optional[str] = None  # Optional transformation: "first", "all", "join", "file_content"
    priority: int = 1  # Higher = checked first


@dataclass
class MappingResult:
    """Result of parameter mapping attempt"""
    success: bool
    value: Any = None
    source: str = ""  # Where the value came from
    confidence: float = 1.0
    error: str = ""


class ParameterMapper:
    """
    Automatic parameter mapper using patterns and schemas.

    Key Design Principles:
    1. No individual tool configuration - use patterns
    2. Schema-driven validation
    3. Semantic parameter name matching
    4. Context-aware data extraction
    5. Parameter name alias correction (NEW)
    """

    # ========================================================================
    # Parameter Name Aliases - Handle common LLM mistakes
    # Maps wrong_name -> correct_name for specific tools or globally
    # ========================================================================

    PARAMETER_ALIASES = {
        # Global aliases (apply to all tools)
        "_global": {
            "pdb_id": "accession",  # LLM often confuses PDB ID with UniProt accession
            "protein_id": "accession",
            "uniprot_id": "accession",
            "id": "accession",
            "gene_id": "gene",
            "gene_symbol": "gene",
            "species": "organism",
            "tax_id": "organism",
            "filepath": "file_path",
            "filename": "file_path",
            "input_path": "file_path",
        },
        # Tool-specific aliases
        "get_protein_info": {
            "pdb_id": "accession",
            "protein_id": "accession",
            "id": "accession",
        },
        "get_protein_sequence": {
            "pdb_id": "accession",
            "protein_id": "accession",
        },
        "search_by_gene": {
            "gene_name": "gene",
            "gene_symbol": "gene",
        },
        "enrichment_analysis": {
            "genes": "gene_list",
            "gene_names": "gene_list",
        },
        "read_metadata_tool": {
            "path": "file_path",
            "filepath": "file_path",
        },
        "run_pandas_code_tool": {
            "python_code": "code",
            "pandas_code": "code",
        },
    }

    # ========================================================================
    # Pattern Rules - These cover most biomedical tools without per-tool config
    # ========================================================================

    DEFAULT_PATTERNS = [
        # === File Path Patterns (Priority: High) ===
        ParameterPattern(
            pattern=r"^(file_path|filepath|path|input_file|data_file)$",
            data_source="input_files",
            extraction_path="first_file.path",
            priority=10
        ),
        ParameterPattern(
            pattern=r"^(bam_file|bam_path|bam1|bam2)$",
            data_source="input_files",
            extraction_path="*.bam.path",
            priority=10
        ),
        ParameterPattern(
            pattern=r"^(pod5_file|pod5_path)$",
            data_source="input_files",
            extraction_path="*.pod5.path",
            priority=10
        ),
        ParameterPattern(
            pattern=r"^(csv_file|csv_path)$",
            data_source="input_files",
            extraction_path="*.csv.path",
            priority=10
        ),

        # === Sequence Patterns ===
        ParameterPattern(
            pattern=r"^(sequence|seq|protein_sequence|amino_acid_sequence|aa_sequence)$",
            data_source="collected_data",
            extraction_path="UniProt.*.sequence",
            transform="first",
            priority=8
        ),
        ParameterPattern(
            pattern=r"^(sequences|seqs)$",
            data_source="collected_data",
            extraction_path="UniProt.*.sequence",
            transform="all",
            priority=8
        ),
        ParameterPattern(
            pattern=r"^(fasta|fasta_sequence)$",
            data_source="collected_data",
            extraction_path="UniProt.*.sequence",
            transform="fasta",
            priority=8
        ),

        # === Gene/Protein ID Patterns ===
        ParameterPattern(
            pattern=r"^(gene_list|genes|gene_names)$",
            data_source="collected_data",
            extraction_path="*.gene_name",
            transform="all",
            priority=7
        ),
        ParameterPattern(
            pattern=r"^(gene|gene_name|gene_symbol)$",
            data_source="collected_data",
            extraction_path="*.gene_name",
            transform="first",
            priority=7
        ),
        ParameterPattern(
            pattern=r"^(protein_id|uniprot_id|accession|uniprot_accession)$",
            data_source="collected_data",
            extraction_path="UniProt.*.accession",
            transform="first",
            priority=7
        ),
        ParameterPattern(
            pattern=r"^(proteins|protein_list|protein_ids)$",
            data_source="collected_data",
            extraction_path="UniProt.*.accession",
            transform="all",
            priority=7
        ),
        ParameterPattern(
            pattern=r"^(ensembl_id|ensembl)$",
            data_source="collected_data",
            extraction_path="*.ensembl",
            transform="first",
            priority=7
        ),

        # === UniProt Accession Pattern (IMPORTANT: for get_protein_info etc.) ===
        ParameterPattern(
            pattern=r"^(accession|uniprot_accession)$",
            data_source="collected_data",
            extraction_path="UniProt.*.accession",
            transform="first",
            priority=9  # High priority for UniProt tools
        ),

        # === Structure ID Patterns ===
        ParameterPattern(
            pattern=r"^(pdb_id|pdb|structure_id)$",
            data_source="collected_data",
            extraction_path="RCSBPDB.*.pdb_id",
            transform="first",
            priority=7
        ),
        ParameterPattern(
            pattern=r"^(pdb_file|structure_file)$",
            data_source="collected_data",
            extraction_path="ESMFold.*.pdb_path",
            transform="first",
            priority=7
        ),

        # === Organism Patterns ===
        ParameterPattern(
            pattern=r"^(organism|species|organism_code|organism_id)$",
            data_source="entity_analysis",
            extraction_path="primary_entities[type=organism].name",
            transform="first",
            priority=6
        ),
        ParameterPattern(
            pattern=r"^(taxon_id|taxonomy_id|ncbi_taxon)$",
            data_source="entity_analysis",
            extraction_path="primary_entities[type=organism].identifiers.taxon_id",
            transform="first",
            priority=6
        ),

        # === Entity-based Query Patterns (for searching) ===
        ParameterPattern(
            pattern=r"^(query|search_query|term|keyword|search_term)$",
            data_source="entity_analysis",
            extraction_path="primary_entities[0].name",
            transform="first",
            priority=6
        ),
        ParameterPattern(
            pattern=r"^(protein_name|target_name|target)$",
            data_source="entity_analysis",
            extraction_path="primary_entities[type=protein].name",
            transform="first",
            priority=6
        ),
        ParameterPattern(
            pattern=r"^(gene|gene_name|gene_symbol)$",
            data_source="entity_analysis",
            extraction_path="primary_entities[type=gene].name",
            transform="first",
            priority=6
        ),

        # === Pathway/Database Patterns ===
        ParameterPattern(
            pattern=r"^(pathway_id|kegg_pathway|pathway)$",
            data_source="collected_data",
            extraction_path="KEGG.*.pathway_id",
            transform="first",
            priority=6
        ),
        ParameterPattern(
            pattern=r"^(pathway_name|pathway_query)$",
            data_source="entity_analysis",
            extraction_path="primary_entities[type=pathway].name",
            transform="first",
            priority=5
        ),

        # === Domain/Region Patterns ===
        ParameterPattern(
            pattern=r"^(domain|domain_name|region)$",
            data_source="entity_analysis",
            extraction_path="primary_entities[type=domain].name",
            transform="first",
            priority=5
        ),

        # === Numeric Parameters (use defaults) ===
        ParameterPattern(
            pattern=r"^(max_results|limit|top_n|n|count)$",
            data_source="default",
            extraction_path="20",
            priority=3
        ),
        ParameterPattern(
            pattern=r"^(threshold|cutoff|min_score)$",
            data_source="default",
            extraction_path="0.5",
            priority=3
        ),
    ]

    # ========================================================================
    # Organism Code Mappings (Common organisms)
    # ========================================================================

    ORGANISM_CODES = {
        "human": "hsa",
        "homo sapiens": "hsa",
        "mouse": "mmu",
        "mus musculus": "mmu",
        "rat": "rno",
        "rattus norvegicus": "rno",
        "zebrafish": "dre",
        "danio rerio": "dre",
        "fruit fly": "dme",
        "drosophila melanogaster": "dme",
        "yeast": "sce",
        "saccharomyces cerevisiae": "sce",
        "e. coli": "eco",
        "escherichia coli": "eco",
    }

    ORGANISM_TAXON_IDS = {
        "human": "9606",
        "homo sapiens": "9606",
        "mouse": "10090",
        "mus musculus": "10090",
        "rat": "10116",
        "zebrafish": "7955",
        "fruit fly": "7227",
        "yeast": "559292",
        "e. coli": "562",
    }

    def __init__(self, patterns: List[ParameterPattern] = None):
        """
        Initialize parameter mapper with pattern rules.

        Args:
            patterns: Custom patterns (uses DEFAULT_PATTERNS if None)
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        # Sort by priority (highest first)
        self.patterns.sort(key=lambda p: -p.priority)
        self.logger = logging.getLogger(self.__class__.__name__)

    def map_parameters(
        self,
        tool_name: str,
        tool_schema: Dict[str, Any],
        provided_args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Map and validate parameters for a tool call.

        Args:
            tool_name: Name of the tool being called
            tool_schema: Tool's inputSchema from MCP server
            provided_args: Arguments provided by LLM
            context: Available context data:
                - input_files: {file_name: {path, type, ...}}
                - collected_data: {source: [...data...]}
                - entity_analysis: {primary_entities: [...], ...}

        Returns:
            Tuple of (mapped_arguments, warnings)
        """
        warnings = []

        properties = tool_schema.get("properties", {})
        required = tool_schema.get("required", [])

        # === STEP 0: Apply parameter name aliases (NEW) ===
        # This fixes common LLM mistakes like using "pdb_id" instead of "accession"
        corrected_args = self._apply_parameter_aliases(tool_name, provided_args, properties)
        mapped_args = dict(corrected_args)

        self.logger.debug(f"Mapping parameters for {tool_name}")
        self.logger.debug(f"  Schema properties: {list(properties.keys())}")
        self.logger.debug(f"  Required: {required}")
        self.logger.debug(f"  Provided: {list(provided_args.keys())}")

        # Process each parameter in schema
        for param_name, param_schema in properties.items():
            # Skip if already provided
            if param_name in mapped_args and mapped_args[param_name] is not None:
                # Validate and potentially transform provided value
                mapped_args[param_name] = self._validate_and_transform(
                    param_name, mapped_args[param_name], param_schema, context
                )
                continue

            # Try to auto-map missing parameter
            result = self._auto_map_parameter(param_name, param_schema, context)

            if result.success:
                mapped_args[param_name] = result.value
                self.logger.info(
                    f"  Auto-mapped {param_name} = {self._truncate(result.value)} "
                    f"(from {result.source})"
                )
            elif param_name in required:
                # Required but couldn't map
                warnings.append(
                    f"Required parameter '{param_name}' could not be auto-mapped. "
                    f"Schema: {param_schema}. Error: {result.error}"
                )

        # Final validation
        missing_required = [p for p in required if p not in mapped_args or mapped_args[p] is None]
        if missing_required:
            warnings.append(f"Missing required parameters: {missing_required}")

        return mapped_args, warnings

    def _apply_parameter_aliases(
        self,
        tool_name: str,
        provided_args: Dict[str, Any],
        schema_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply parameter name aliases to fix common LLM mistakes.

        For example, if LLM provides "pdb_id" but the tool expects "accession",
        this method will rename the parameter.

        Args:
            tool_name: Name of the tool being called
            provided_args: Original arguments from LLM
            schema_properties: Tool's schema properties

        Returns:
            Corrected arguments dict
        """
        corrected = {}
        schema_param_names = set(schema_properties.keys())

        # Get aliases for this tool and global aliases
        tool_aliases = self.PARAMETER_ALIASES.get(tool_name, {})
        global_aliases = self.PARAMETER_ALIASES.get("_global", {})

        for param_name, value in provided_args.items():
            # If parameter is already in schema, keep it
            if param_name in schema_param_names:
                corrected[param_name] = value
                continue

            # Check tool-specific alias first
            if param_name in tool_aliases:
                correct_name = tool_aliases[param_name]
                if correct_name in schema_param_names:
                    self.logger.info(
                        f"  Alias correction: {param_name} → {correct_name} "
                        f"(tool-specific for {tool_name})"
                    )
                    corrected[correct_name] = value
                    continue

            # Check global alias
            if param_name in global_aliases:
                correct_name = global_aliases[param_name]
                if correct_name in schema_param_names:
                    self.logger.info(
                        f"  Alias correction: {param_name} → {correct_name} (global)"
                    )
                    corrected[correct_name] = value
                    continue

            # No alias found, keep original (will likely fail validation)
            corrected[param_name] = value
            self.logger.warning(
                f"  Unknown parameter '{param_name}' not in schema "
                f"and no alias found. Schema expects: {list(schema_param_names)}"
            )

        return corrected

    def _auto_map_parameter(
        self,
        param_name: str,
        param_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> MappingResult:
        """
        Attempt to automatically map a parameter using patterns.

        Args:
            param_name: Name of the parameter
            param_schema: Schema for this parameter
            context: Available context data

        Returns:
            MappingResult with success status and value
        """
        # Try pattern matching first
        for pattern in self.patterns:
            if re.match(pattern.pattern, param_name, re.IGNORECASE):
                result = self._extract_from_pattern(pattern, param_schema, context)
                if result.success:
                    return result

        # Try semantic matching based on description
        description = param_schema.get("description", "").lower()
        result = self._semantic_match(param_name, description, param_schema, context)
        if result.success:
            return result

        # Try type-based inference
        result = self._type_based_inference(param_name, param_schema, context)
        if result.success:
            return result

        return MappingResult(success=False, error="No matching pattern or semantic match found")

    def _extract_from_pattern(
        self,
        pattern: ParameterPattern,
        param_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> MappingResult:
        """Extract value based on pattern rule"""
        try:
            source_data = context.get(pattern.data_source, {})

            # Handle default values
            if pattern.data_source == "default":
                value = self._parse_default(pattern.extraction_path, param_schema)
                return MappingResult(
                    success=True,
                    value=value,
                    source=f"default:{pattern.extraction_path}"
                )

            # Handle input_files specially
            if pattern.data_source == "input_files":
                value = self._extract_from_input_files(pattern, source_data)
                if value is not None:
                    return MappingResult(
                        success=True,
                        value=value,
                        source="input_files"
                    )
                return MappingResult(success=False, error="No matching input file found")

            # Handle collected_data extraction
            if pattern.data_source == "collected_data":
                value = self._extract_from_collected_data(
                    pattern.extraction_path,
                    source_data,
                    pattern.transform
                )
                if value is not None:
                    return MappingResult(
                        success=True,
                        value=value,
                        source=f"collected_data:{pattern.extraction_path}"
                    )
                return MappingResult(success=False, error="No matching data in collected_data")

            # Handle entity_analysis extraction
            if pattern.data_source == "entity_analysis":
                value = self._extract_from_entity_analysis(
                    pattern.extraction_path,
                    source_data,
                    pattern.transform
                )
                if value is not None:
                    return MappingResult(
                        success=True,
                        value=value,
                        source=f"entity_analysis:{pattern.extraction_path}"
                    )
                return MappingResult(success=False, error="No matching data in entity_analysis")

            return MappingResult(success=False, error=f"Unknown data source: {pattern.data_source}")

        except Exception as e:
            return MappingResult(success=False, error=str(e))

    def _extract_from_input_files(
        self,
        pattern: ParameterPattern,
        input_files: Dict[str, Any]
    ) -> Optional[str]:
        """Extract file path from input_files based on pattern"""
        if not input_files:
            return None

        extraction = pattern.extraction_path

        # Handle "first_file.path" - return first file's path
        if extraction == "first_file.path":
            for file_name, file_info in input_files.items():
                if isinstance(file_info, dict) and "path" in file_info:
                    return file_info["path"]
            return None

        # Handle "*.{extension}.path" - find file by extension
        if extraction.startswith("*.") and extraction.endswith(".path"):
            ext = extraction[2:-5]  # Extract extension
            for file_name, file_info in input_files.items():
                if isinstance(file_info, dict):
                    file_type = file_info.get("type", "").lower()
                    if file_type == ext or file_name.lower().endswith(f".{ext}"):
                        return file_info.get("path")
            return None

        return None

    def _extract_from_collected_data(
        self,
        extraction_path: str,
        collected_data: Dict[str, Any],
        transform: Optional[str]
    ) -> Any:
        """
        Extract value from collected_data using path pattern.

        Path format: "Source.*.field" or "*.field"
        Examples:
            - "UniProt.*.sequence" → sequences from UniProt results
            - "*.gene_name" → gene names from any source
        """
        if not collected_data:
            return None

        values = []
        parts = extraction_path.split(".")

        # Determine which sources to search
        if parts[0] == "*":
            sources = list(collected_data.keys())
            field_path = parts[1:]
        else:
            sources = [parts[0]] if parts[0] in collected_data else []
            field_path = parts[1:]

        # Extract values from each source
        for source in sources:
            source_data = collected_data.get(source, [])
            if isinstance(source_data, list):
                for item in source_data:
                    value = self._get_nested_value(item, field_path)
                    if value is not None:
                        values.append(value)
            elif isinstance(source_data, dict):
                value = self._get_nested_value(source_data, field_path)
                if value is not None:
                    values.append(value)

        if not values:
            return None

        # Apply transform
        if transform == "first":
            return values[0] if values else None
        elif transform == "all":
            return values
        elif transform == "join":
            return ",".join(str(v) for v in values)
        elif transform == "fasta":
            # Format as FASTA
            fasta_lines = []
            for i, seq in enumerate(values):
                fasta_lines.append(f">sequence_{i+1}")
                fasta_lines.append(str(seq))
            return "\n".join(fasta_lines)
        else:
            return values[0] if len(values) == 1 else values

    def _extract_from_entity_analysis(
        self,
        extraction_path: str,
        entity_analysis: Dict[str, Any],
        transform: Optional[str]
    ) -> Any:
        """
        Extract value from entity_analysis with enhanced support.

        Supported patterns:
        - primary_entities[type=X].field - get field from first entity of type X
        - primary_entities[type=X].identifiers.key - get nested identifier
        - primary_entities[0].field - get field from first entity
        - context_refinements.key - get from context refinements
        - data_requirements[type=X].source - get source from data requirement
        """
        if not entity_analysis:
            return None

        # Handle "primary_entities[type=X].identifiers.key" pattern (nested)
        match = re.match(r"primary_entities\[type=(\w+)\]\.identifiers\.(\w+)", extraction_path)
        if match:
            entity_type, identifier_key = match.groups()
            entities = entity_analysis.get("primary_entities", [])
            for entity in entities:
                if entity.get("type") == entity_type:
                    identifiers = entity.get("identifiers", {})
                    if identifier_key in identifiers:
                        return identifiers[identifier_key]
            # Fallback: try to infer from name
            if entity_type == "organism" and identifier_key == "taxon_id":
                for entity in entities:
                    if entity.get("type") == "organism":
                        return self._get_organism_taxon_id(entity.get("name", ""))
            return None

        # Handle "primary_entities[type=X].field" pattern
        match = re.match(r"primary_entities\[type=(\w+)\]\.(\w+)", extraction_path)
        if match:
            entity_type, field = match.groups()
            entities = entity_analysis.get("primary_entities", [])

            # First, look for exact type match
            for entity in entities:
                if entity.get("type") == entity_type:
                    value = entity.get(field)
                    # Convert organism name to code if needed
                    if field == "name" and entity_type == "organism":
                        return self._get_organism_code(value)
                    return value

            # Fallback for organism: default to human for biomedical research
            if entity_type == "organism" and field == "name":
                self.logger.info("  No organism found in entities, defaulting to 'hsa' (human)")
                return "hsa"

            # Fallback: try first entity if looking for gene/protein (often interchangeable)
            if entity_type in ["gene", "protein"] and field == "name" and entities:
                for entity in entities:
                    if entity.get("type") in ["gene", "protein"]:
                        return entity.get(field)

            return None

        # Handle "context_refinements.key" pattern
        match = re.match(r"context_refinements\.(\w+)", extraction_path)
        if match:
            key = match.group(1)
            refinements = entity_analysis.get("context_refinements", {})
            return refinements.get(key)

        # Handle "data_requirements[type=X].source" pattern
        match = re.match(r"data_requirements\[type=(\w+)\]\.(\w+)", extraction_path)
        if match:
            req_type, field = match.groups()
            data_reqs = entity_analysis.get("data_requirements", [])
            for req in data_reqs:
                if req.get("type") == req_type:
                    return req.get(field)
            return None

        # Handle "primary_entities[0].field" pattern
        match = re.match(r"primary_entities\[(\d+)\]\.(\w+)", extraction_path)
        if match:
            index, field = match.groups()
            entities = entity_analysis.get("primary_entities", [])
            if int(index) < len(entities):
                return entities[int(index)].get(field)
            return None

        # Direct field access
        return entity_analysis.get(extraction_path)

    def _get_nested_value(self, data: Any, path: List[str]) -> Any:
        """Get nested value from dict/list using path"""
        current = data
        for part in path:
            if part == "*":
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            else:
                return None
            if current is None:
                return None
        return current

    def _get_organism_code(self, organism_name: str) -> str:
        """Convert organism name to KEGG organism code"""
        if not organism_name:
            return "hsa"  # Default to human

        name_lower = organism_name.lower()

        # Check direct mapping
        if name_lower in self.ORGANISM_CODES:
            return self.ORGANISM_CODES[name_lower]

        # Check partial match
        for key, code in self.ORGANISM_CODES.items():
            if key in name_lower or name_lower in key:
                return code

        # Check if it's already a code
        if len(organism_name) == 3 and organism_name.isalpha():
            return organism_name.lower()

        return "hsa"  # Default

    def _get_organism_taxon_id(self, organism_name: str) -> str:
        """Convert organism name to NCBI Taxonomy ID"""
        if not organism_name:
            return "9606"  # Default to human

        name_lower = organism_name.lower()

        # Check direct mapping
        if name_lower in self.ORGANISM_TAXON_IDS:
            return self.ORGANISM_TAXON_IDS[name_lower]

        # Check partial match
        for key, taxon_id in self.ORGANISM_TAXON_IDS.items():
            if key in name_lower or name_lower in key:
                return taxon_id

        # Check if it's already a taxon ID (numeric)
        if organism_name.isdigit():
            return organism_name

        return "9606"  # Default to human

    def _semantic_match(
        self,
        param_name: str,
        description: str,
        param_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> MappingResult:
        """Try to match parameter based on semantic analysis of name/description"""

        # Keywords that suggest specific data types
        keywords = {
            "sequence": ["sequence", "amino acid", "protein sequence", "aa sequence"],
            "gene": ["gene", "gene name", "gene symbol", "gene id"],
            "protein": ["protein", "uniprot", "accession"],
            "file": ["file", "path", "input"],
            "organism": ["organism", "species", "taxon"],
        }

        combined_text = f"{param_name} {description}".lower()

        for data_type, kws in keywords.items():
            if any(kw in combined_text for kw in kws):
                # Found a semantic match, try to extract
                if data_type == "sequence":
                    value = self._extract_from_collected_data(
                        "UniProt.*.sequence",
                        context.get("collected_data", {}),
                        "first"
                    )
                    if value:
                        return MappingResult(success=True, value=value, source="semantic:sequence")

                elif data_type == "gene":
                    value = self._extract_from_collected_data(
                        "*.gene_name",
                        context.get("collected_data", {}),
                        "first" if "list" not in param_name.lower() else "all"
                    )
                    if value:
                        return MappingResult(success=True, value=value, source="semantic:gene")

                elif data_type == "file":
                    input_files = context.get("input_files", {})
                    for file_name, file_info in input_files.items():
                        if isinstance(file_info, dict) and "path" in file_info:
                            return MappingResult(
                                success=True,
                                value=file_info["path"],
                                source="semantic:file"
                            )

                elif data_type == "organism":
                    entity_analysis = context.get("entity_analysis", {})
                    for entity in entity_analysis.get("primary_entities", []):
                        if entity.get("type") == "organism":
                            code = self._get_organism_code(entity.get("name", ""))
                            return MappingResult(success=True, value=code, source="semantic:organism")

        return MappingResult(success=False, error="No semantic match")

    def _type_based_inference(
        self,
        param_name: str,
        param_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> MappingResult:
        """Infer parameter value based on schema type"""
        param_type = param_schema.get("type", "string")

        # Handle enum types
        if "enum" in param_schema:
            enum_values = param_schema["enum"]
            if enum_values:
                return MappingResult(
                    success=True,
                    value=enum_values[0],  # Use first enum value as default
                    source="type:enum_default",
                    confidence=0.5
                )

        # Handle boolean with default
        if param_type == "boolean":
            default = param_schema.get("default", False)
            return MappingResult(success=True, value=default, source="type:boolean_default")

        # Handle integer with reasonable defaults
        if param_type == "integer":
            if "max" in param_name.lower() or "limit" in param_name.lower():
                return MappingResult(success=True, value=20, source="type:int_default")

        return MappingResult(success=False, error="No type-based inference available")

    def _validate_and_transform(
        self,
        param_name: str,
        value: Any,
        param_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Validate and potentially transform a provided value"""
        param_type = param_schema.get("type", "string")

        # Convert file path patterns to actual paths
        if isinstance(value, str):
            # Check if it's a file reference that needs resolution
            if value.startswith("$input_files."):
                file_key = value.split(".", 1)[1]
                input_files = context.get("input_files", {})
                if file_key in input_files:
                    return input_files[file_key].get("path", value)

            # Convert relative to absolute path if it looks like a file
            if any(value.endswith(ext) for ext in [".csv", ".bam", ".pod5", ".pdb", ".json"]):
                input_files = context.get("input_files", {})
                for file_name, file_info in input_files.items():
                    if file_name == value or value.endswith(file_name):
                        return file_info.get("path", value)

        # Type coercion
        if param_type == "array" and isinstance(value, str):
            # Try to convert string to array
            if "," in value:
                return [v.strip() for v in value.split(",")]
            return [value]

        return value

    def _parse_default(self, default_str: str, param_schema: Dict[str, Any]) -> Any:
        """Parse default value string to appropriate type"""
        param_type = param_schema.get("type", "string")

        if param_type == "integer":
            return int(default_str)
        elif param_type == "number":
            return float(default_str)
        elif param_type == "boolean":
            return default_str.lower() in ("true", "1", "yes")
        else:
            return default_str

    def _truncate(self, value: Any, max_len: int = 50) -> str:
        """Truncate value for logging"""
        s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return s


# ============================================================================
# Integration with ToolExecutor
# ============================================================================

def create_mapper_context(
    input_files: Dict[str, Any] = None,
    collected_data: Dict[str, Any] = None,
    entity_analysis: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create context dict for ParameterMapper.

    Args:
        input_files: Input data files mapping {name: {path, type, ...}}
        collected_data: Data from Chain 1 collection
        entity_analysis: Entity analysis from Stage 1

    Returns:
        Context dict for use with ParameterMapper.map_parameters()
    """
    return {
        "input_files": input_files or {},
        "collected_data": collected_data or {},
        "entity_analysis": entity_analysis or {}
    }
