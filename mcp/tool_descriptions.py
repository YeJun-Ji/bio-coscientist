"""
Standardized Tool Descriptions for MCP Servers

This module defines the standard format and descriptions for all MCP tools.
The format enables LLM to automatically select appropriate tools based on
entity types and use cases without hard-coded mappings.

STANDARD FORMAT:
"[Action description]. USE FOR: [use cases]. ENTITY TYPES: [types]. DATA FLOW: [produces/requires]."

Categories:
- collection: Data gathering tools
- analysis: Computational analysis tools
- design: De novo design tools (GPU)
"""

# ============================================================================
# TOOL DESCRIPTION STANDARDS
# ============================================================================

TOOL_DESCRIPTIONS = {
    # ========================================================================
    # STRING DB - PPI Data Collection
    # ========================================================================
    "stringdb": {
        "get_protein_network": (
            "Get protein-protein interaction network from STRING database. "
            "USE FOR: PPI network construction, interactor discovery, pathway mapping. "
            "ENTITY TYPES: protein, gene. "
            "DATA FLOW: Produces interaction edges with confidence scores for NetworkX analysis."
        ),
        "get_interaction_partners": (
            "Get interaction partners for a protein with confidence scores. "
            "USE FOR: Interactor identification, hub protein discovery. "
            "ENTITY TYPES: protein, gene. "
            "DATA FLOW: Produces partner list that can be used for network analysis or enrichment."
        ),
        "get_enrichment_analysis": (
            "Perform GO/KEGG functional enrichment on protein list. "
            "USE FOR: Functional annotation, pathway enrichment, biological process identification. "
            "ENTITY TYPES: protein, gene, pathway. "
            "DATA FLOW: Requires protein list, produces enriched terms and pathways."
        ),
        "get_network_image": (
            "Get network visualization image URL. "
            "USE FOR: Network visualization, figure generation. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires protein list, produces image URL."
        ),
    },

    # ========================================================================
    # NetworkX - Network Analysis
    # ========================================================================
    "networkx": {
        "build_network": (
            "Build protein interaction network from STRING data for analysis. "
            "USE FOR: Network construction, graph analysis preparation. "
            "ENTITY TYPES: protein, gene, network. "
            "DATA FLOW: Requires protein list, produces NetworkX graph object for analysis."
        ),
        "find_hub_proteins": (
            "Identify hub proteins with high connectivity in the network. "
            "USE FOR: Key node identification, drug target prioritization, network centrality. "
            "ENTITY TYPES: protein, network. "
            "DATA FLOW: Requires network data (from STRING), produces ranked hub proteins."
        ),
        "find_communities": (
            "Detect protein communities/modules in the network. "
            "USE FOR: Module detection, functional clustering, pathway grouping. "
            "ENTITY TYPES: protein, pathway, network. "
            "DATA FLOW: Requires network data, produces community assignments."
        ),
        "calculate_centrality": (
            "Calculate network centrality metrics (degree, betweenness, closeness). "
            "USE FOR: Node importance ranking, bottleneck identification. "
            "ENTITY TYPES: protein, network. "
            "DATA FLOW: Requires network data, produces centrality scores per node."
        ),
        "shortest_path": (
            "Find shortest path between two proteins in the network. "
            "USE FOR: Pathway discovery, signal transduction analysis. "
            "ENTITY TYPES: protein, pathway. "
            "DATA FLOW: Requires network and two protein IDs, produces path."
        ),
    },

    # ========================================================================
    # RCSB PDB - Structure Data
    # ========================================================================
    "rcsbpdb": {
        "download_pdb": (
            "Download protein structure from RCSB PDB database. "
            "USE FOR: Structure retrieval, structural analysis preparation. "
            "ENTITY TYPES: protein, structure. "
            "DATA FLOW: Produces PDB file content for docking, visualization, or design."
        ),
        "search_structures": (
            "Search PDB for structures by protein name, sequence, or keywords. "
            "USE FOR: Structure discovery, template finding. "
            "ENTITY TYPES: protein, structure. "
            "DATA FLOW: Produces list of PDB IDs for subsequent download."
        ),
        "get_binding_sites": (
            "Analyze binding sites and ligand interactions in a structure. "
            "USE FOR: Binding site identification, drug design target analysis, hotspot residues. "
            "ENTITY TYPES: protein, domain, compound. "
            "DATA FLOW: Requires PDB ID, produces binding site residues and ligand info."
        ),
        "get_structure_info": (
            "Get metadata and annotations for a PDB structure. "
            "USE FOR: Structure annotation, resolution check, method verification. "
            "ENTITY TYPES: protein, structure. "
            "DATA FLOW: Produces structure metadata (resolution, method, chains)."
        ),
    },

    # ========================================================================
    # InterPro - Domain Analysis
    # ========================================================================
    "interpro": {
        "analyze_domains": (
            "Analyze protein domains and families using InterPro database. "
            "USE FOR: Domain identification, functional annotation, family classification. "
            "ENTITY TYPES: protein, domain. "
            "DATA FLOW: Requires protein sequence or ID, produces domain annotations."
        ),
        "get_domain_architecture": (
            "Get complete domain architecture of a protein. "
            "USE FOR: Protein structure understanding, binding site context, multi-domain analysis. "
            "ENTITY TYPES: protein, domain. "
            "DATA FLOW: Requires UniProt ID, produces domain layout with positions."
        ),
        "search_by_domain": (
            "Find proteins containing a specific domain. "
            "USE FOR: Homolog discovery, protein family analysis. "
            "ENTITY TYPES: protein, domain. "
            "DATA FLOW: Requires InterPro domain ID, produces protein list."
        ),
    },

    # ========================================================================
    # gProfiler - Enrichment Analysis
    # ========================================================================
    "gprofiler": {
        "enrichment_analysis": (
            "Perform functional enrichment analysis (GO, KEGG, Reactome) on gene/protein list. "
            "USE FOR: Pathway enrichment, GO term analysis, functional annotation of gene sets. "
            "ENTITY TYPES: gene, protein, pathway. "
            "DATA FLOW: Requires gene/protein list, produces enriched terms with p-values."
        ),
        "convert_gene_ids": (
            "Convert gene/protein identifiers between databases (Ensembl, UniProt, Symbol). "
            "USE FOR: ID mapping, cross-database integration. "
            "ENTITY TYPES: gene, protein. "
            "DATA FLOW: Requires gene list and source/target DB, produces mapped IDs."
        ),
    },

    # ========================================================================
    # OpenTargets - Druggability
    # ========================================================================
    "opentargets": {
        "search_target": (
            "Search for drug targets by gene name or disease. "
            "USE FOR: Target discovery, drug-target relationship lookup. "
            "ENTITY TYPES: gene, protein, disease, compound. "
            "DATA FLOW: Produces target info including known drugs and clinical status."
        ),
        "assess_druggability": (
            "Evaluate druggability of a protein target. "
            "USE FOR: Target prioritization, drug development feasibility assessment. "
            "ENTITY TYPES: protein, gene. "
            "DATA FLOW: Requires Ensembl ID, produces druggability scores and tractability."
        ),
        "get_associated_drugs": (
            "Get drugs associated with a target or disease. "
            "USE FOR: Existing drug identification, repurposing opportunities. "
            "ENTITY TYPES: protein, disease, compound. "
            "DATA FLOW: Produces drug list with mechanism and clinical phase."
        ),
        "get_disease_associations": (
            "Get disease associations for a target. "
            "USE FOR: Disease relevance assessment, indication discovery. "
            "ENTITY TYPES: protein, gene, disease. "
            "DATA FLOW: Requires target ID, produces disease associations with evidence."
        ),
    },

    # ========================================================================
    # IEDB - Immunogenicity
    # ========================================================================
    "iedb": {
        "predict_mhc_binding": (
            "Predict MHC class I binding for peptides. "
            "USE FOR: Immunogenicity assessment, epitope prediction, vaccine design. "
            "ENTITY TYPES: protein, peptide. "
            "DATA FLOW: Requires sequence and HLA alleles, produces binding predictions."
        ),
        "predict_mhc_ii_binding": (
            "Predict MHC class II binding for peptides. "
            "USE FOR: T-helper epitope prediction, immunogenicity screening. "
            "ENTITY TYPES: protein, peptide. "
            "DATA FLOW: Requires sequence and HLA-DR alleles, produces binding predictions."
        ),
        "scan_protein": (
            "Scan entire protein sequence for immunogenic epitopes. "
            "USE FOR: Immunogenicity hotspot identification, therapeutic protein assessment. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires full protein sequence, produces epitope map."
        ),
    },

    # ========================================================================
    # Vina - Molecular Docking
    # ========================================================================
    "vina": {
        "dock_ligand": (
            "Perform molecular docking of small molecule to protein. "
            "USE FOR: Virtual screening, binding pose prediction, drug-target interaction. "
            "ENTITY TYPES: protein, compound. "
            "DATA FLOW: Requires receptor PDB and ligand SMILES/SDF, produces docked poses."
        ),
        "calculate_binding_affinity": (
            "Calculate binding affinity score for docked complex. "
            "USE FOR: Affinity ranking, lead optimization. "
            "ENTITY TYPES: protein, compound. "
            "DATA FLOW: Requires docked pose, produces binding energy (kcal/mol)."
        ),
        "prepare_receptor": (
            "Prepare protein receptor for docking (add hydrogens, charges). "
            "USE FOR: Docking preparation. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires PDB, produces PDBQT file."
        ),
    },

    # ========================================================================
    # ProteinMPNN - Sequence Design (GPU)
    # ========================================================================
    "proteinmpnn": {
        "design_sequence": (
            "Design protein sequence for a given backbone structure. "
            "USE FOR: Inverse folding, sequence optimization, de novo protein design. "
            "ENTITY TYPES: protein, structure. "
            "DATA FLOW: Requires backbone PDB, produces designed sequences with scores. "
            "REQUIRES: GPU (8GB+)."
        ),
        "design_with_fixed_positions": (
            "Design sequence with some positions fixed (e.g., binding site). "
            "USE FOR: Constrained design, binding site preservation, scaffold optimization. "
            "ENTITY TYPES: protein, domain. "
            "DATA FLOW: Requires backbone PDB and fixed residue positions. "
            "REQUIRES: GPU (8GB+)."
        ),
        "score_sequence": (
            "Score sequence-structure compatibility. "
            "USE FOR: Design validation, sequence quality assessment. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires sequence and backbone, produces compatibility score. "
            "REQUIRES: GPU (8GB+)."
        ),
    },

    # ========================================================================
    # RFdiffusion - Backbone Design (GPU)
    # ========================================================================
    "rfdiffusion": {
        "design_binder": (
            "Design de novo protein binder backbone for a target. "
            "USE FOR: Binder design, PPI inhibitor development, therapeutic protein design. "
            "ENTITY TYPES: protein, domain. "
            "DATA FLOW: Requires target PDB and hotspot residues, produces binder backbone. "
            "REQUIRES: GPU (16GB+)."
        ),
        "unconditional_design": (
            "Generate novel protein backbones without constraints. "
            "USE FOR: Novel scaffold generation, protein engineering. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Produces novel backbone structures. "
            "REQUIRES: GPU (16GB+)."
        ),
        "scaffold_conditioning": (
            "Design protein with scaffold constraints (partial structure). "
            "USE FOR: Constrained backbone design, loop modeling, domain insertion. "
            "ENTITY TYPES: protein, domain. "
            "DATA FLOW: Requires scaffold PDB and contig specification. "
            "REQUIRES: GPU (16GB+)."
        ),
    },

    # ========================================================================
    # ColabFold - Structure Prediction (GPU)
    # ========================================================================
    "colabfold": {
        "predict_structure": (
            "Predict 3D structure for a single protein sequence using AlphaFold2. "
            "USE FOR: Structure prediction, fold verification, model generation. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires sequence, produces PDB with pLDDT confidence. "
            "REQUIRES: GPU (24GB+)."
        ),
        "predict_complex": (
            "Predict structure of protein complex (multimer). "
            "USE FOR: Complex structure prediction, PPI interface modeling, binder validation. "
            "ENTITY TYPES: protein, complex. "
            "DATA FLOW: Requires multiple sequences, produces complex PDB with ipTM score. "
            "REQUIRES: GPU (24GB+)."
        ),
        "predict_binder_complex": (
            "Predict binder-target complex structure for validation. "
            "USE FOR: Binder design validation, interface quality assessment. "
            "ENTITY TYPES: protein, complex. "
            "DATA FLOW: Requires binder and target sequences, produces complex with confidence. "
            "REQUIRES: GPU (24GB+)."
        ),
        "get_confidence_metrics": (
            "Get explanation of ColabFold confidence metrics (pLDDT, pTM, ipTM). "
            "USE FOR: Result interpretation, quality thresholds. "
            "ENTITY TYPES: N/A. "
            "DATA FLOW: Produces metric explanations."
        ),
    },

    # ========================================================================
    # ESMFold - Fast Structure Prediction
    # ========================================================================
    "esmfold": {
        "predict_structure": (
            "Predict protein structure using ESMFold (fast, single sequence). "
            "USE FOR: Quick structure prediction, initial fold assessment, screening. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires sequence, produces PDB with pLDDT. Faster than ColabFold."
        ),
        "get_plddt_scores": (
            "Get per-residue pLDDT confidence scores from prediction. "
            "USE FOR: Structure quality assessment, disordered region identification. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Produces confidence scores per residue."
        ),
    },

    # ========================================================================
    # BLAST - Sequence Similarity
    # ========================================================================
    "blast": {
        "blastp": (
            "Search for similar protein sequences using BLASTP. "
            "USE FOR: Homolog discovery, off-target analysis, evolutionary analysis. "
            "ENTITY TYPES: protein, gene. "
            "DATA FLOW: Requires protein sequence, produces similar sequences with E-values."
        ),
        "blastn": (
            "Search for similar nucleotide sequences using BLASTN. "
            "USE FOR: Gene homolog discovery, sequence conservation analysis. "
            "ENTITY TYPES: gene. "
            "DATA FLOW: Requires nucleotide sequence, produces similar sequences."
        ),
    },

    # ========================================================================
    # Rosetta - Protein Modeling
    # ========================================================================
    "rosetta": {
        "calculate_energy": (
            "Calculate Rosetta energy score for a protein structure. "
            "USE FOR: Structure quality assessment, stability prediction. "
            "ENTITY TYPES: protein, structure. "
            "DATA FLOW: Requires PDB, produces energy breakdown."
        ),
        "relax_structure": (
            "Energy minimize (relax) a protein structure. "
            "USE FOR: Structure refinement, pre-docking preparation. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires PDB, produces relaxed PDB."
        ),
        "dock_proteins": (
            "Perform protein-protein docking. "
            "USE FOR: Complex modeling, interface prediction, PPI analysis. "
            "ENTITY TYPES: protein, complex. "
            "DATA FLOW: Requires two PDBs, produces docked complex with scores."
        ),
        "calculate_ddg": (
            "Calculate mutation effects (ddG) on stability. "
            "USE FOR: Mutation analysis, variant effect prediction. "
            "ENTITY TYPES: protein, variant. "
            "DATA FLOW: Requires PDB and mutations, produces ddG values."
        ),
    },

    # ========================================================================
    # Foldseek - Structure Similarity
    # ========================================================================
    "foldseek": {
        "search_structure": (
            "Search for structurally similar proteins using Foldseek. "
            "USE FOR: Structural homolog discovery, fold classification, remote homology. "
            "ENTITY TYPES: protein, structure. "
            "DATA FLOW: Requires PDB or structure, produces similar structures with TM-scores."
        ),
        "cluster_structures": (
            "Cluster protein structures by similarity. "
            "USE FOR: Structure family analysis, representative selection. "
            "ENTITY TYPES: protein, structure. "
            "DATA FLOW: Requires structure set, produces clusters."
        ),
    },

    # ========================================================================
    # MSA - Multiple Sequence Alignment
    # ========================================================================
    "msa": {
        "align_sequences": (
            "Perform multiple sequence alignment using Clustal Omega. "
            "USE FOR: Conservation analysis, phylogenetics, homolog comparison. "
            "ENTITY TYPES: protein, gene. "
            "DATA FLOW: Requires multiple sequences, produces alignment."
        ),
        "calculate_conservation": (
            "Calculate conservation scores from alignment. "
            "USE FOR: Conserved residue identification, functional site prediction. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires alignment, produces per-position conservation scores."
        ),
    },

    # ========================================================================
    # ChEMBL - Drug Data
    # ========================================================================
    "chembl": {
        "search_target": (
            "Search ChEMBL for target by name or ID. "
            "USE FOR: Drug target lookup, bioactivity data retrieval. "
            "ENTITY TYPES: protein, gene. "
            "DATA FLOW: Produces target info and ChEMBL ID."
        ),
        "get_bioactivities": (
            "Get bioactivity data for a target (IC50, Ki, etc.). "
            "USE FOR: Known inhibitor discovery, SAR analysis, lead finding. "
            "ENTITY TYPES: protein, compound. "
            "DATA FLOW: Requires ChEMBL target ID, produces compound activities."
        ),
        "get_compound_info": (
            "Get compound information from ChEMBL. "
            "USE FOR: Drug property lookup, structure retrieval. "
            "ENTITY TYPES: compound. "
            "DATA FLOW: Requires ChEMBL compound ID, produces properties and SMILES."
        ),
    },

    # ========================================================================
    # KEGG - Pathway Data (TypeScript - reference only)
    # ========================================================================
    "kegg": {
        "search_pathway": (
            "Search KEGG pathway database by keyword. "
            "USE FOR: Pathway discovery, biological process lookup. "
            "ENTITY TYPES: pathway. "
            "DATA FLOW: Produces pathway IDs and names."
        ),
        "get_pathway_info": (
            "Get detailed pathway information including genes and compounds. "
            "USE FOR: Pathway analysis, gene-pathway mapping. "
            "ENTITY TYPES: pathway, gene, compound. "
            "DATA FLOW: Requires KEGG pathway ID, produces pathway details."
        ),
        "find_pathways_by_gene": (
            "Find pathways containing a specific gene. "
            "USE FOR: Gene function context, pathway involvement analysis. "
            "ENTITY TYPES: gene, pathway. "
            "DATA FLOW: Requires gene symbol, produces pathway list."
        ),
        "get_compound_info": (
            "Get compound information from KEGG. "
            "USE FOR: Drug/metabolite lookup. "
            "ENTITY TYPES: compound. "
            "DATA FLOW: Requires KEGG compound ID, produces properties."
        ),
    },

    # ========================================================================
    # UniProt - Protein Data (TypeScript - reference only)
    # ========================================================================
    "uniprot": {
        "search_proteins": (
            "Search UniProt for proteins by name, gene, or keywords. "
            "USE FOR: Protein discovery, annotation lookup. "
            "ENTITY TYPES: protein, gene. "
            "DATA FLOW: Produces UniProt entries with IDs."
        ),
        "get_protein_info": (
            "Get detailed protein information from UniProt. "
            "USE FOR: Protein annotation, sequence retrieval, function lookup. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires UniProt ID, produces full protein record."
        ),
        "get_protein_sequence": (
            "Get protein sequence from UniProt. "
            "USE FOR: Sequence retrieval for prediction/alignment. "
            "ENTITY TYPES: protein. "
            "DATA FLOW: Requires UniProt ID, produces FASTA sequence."
        ),
        "get_protein_features": (
            "Get protein features (domains, PTMs, variants). "
            "USE FOR: Domain analysis, binding site lookup, variant annotation. "
            "ENTITY TYPES: protein, domain. "
            "DATA FLOW: Requires UniProt ID, produces feature annotations."
        ),
    },

    # ========================================================================
    # NCBI - Literature & Gene Data (TypeScript - reference only)
    # ========================================================================
    "ncbi": {
        "search_pubmed": (
            "Search PubMed for scientific literature. "
            "USE FOR: Literature review, evidence gathering, research context. "
            "ENTITY TYPES: literature. "
            "DATA FLOW: Produces article list with PMIDs."
        ),
        "get_gene_info": (
            "Get gene information from NCBI Gene database. "
            "USE FOR: Gene annotation, function lookup. "
            "ENTITY TYPES: gene. "
            "DATA FLOW: Requires gene symbol or ID, produces gene record."
        ),
        "search_gene": (
            "Search NCBI Gene database. "
            "USE FOR: Gene discovery, ID mapping. "
            "ENTITY TYPES: gene. "
            "DATA FLOW: Produces gene entries."
        ),
    },

    # ========================================================================
    # Nanopore - Sequencing Analysis
    # ========================================================================
    "nanopore": {
        "analyze_polya": (
            "Analyze poly(A) tail length from Nanopore sequencing data. "
            "USE FOR: Poly(A) analysis, mRNA stability assessment. "
            "ENTITY TYPES: gene, assay. "
            "DATA FLOW: Requires POD5/FAST5 files, produces poly(A) lengths."
        ),
        "detect_modifications": (
            "Detect base modifications from Nanopore signal. "
            "USE FOR: Epitranscriptomics, modified base detection. "
            "ENTITY TYPES: gene, assay. "
            "DATA FLOW: Requires Nanopore reads, produces modification calls."
        ),
    },

    # ========================================================================
    # Pandas Analysis - Data Processing
    # ========================================================================
    "pandas": {
        "analyze_csv": (
            "Analyze CSV/tabular data with statistical summaries. "
            "USE FOR: Data exploration, statistical analysis, data validation. "
            "ENTITY TYPES: data, assay. "
            "DATA FLOW: Requires CSV file path, produces statistical summary."
        ),
        "filter_data": (
            "Filter and transform tabular data based on conditions. "
            "USE FOR: Data preprocessing, subset selection. "
            "ENTITY TYPES: data. "
            "DATA FLOW: Requires data and filter conditions, produces filtered data."
        ),
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_tool_description(server: str, tool: str) -> str:
    """Get standardized description for a tool."""
    server_tools = TOOL_DESCRIPTIONS.get(server, {})
    return server_tools.get(tool, f"Tool {tool} from {server} server.")


def get_tools_by_entity_type(entity_type: str) -> list:
    """Get all tools that work with a specific entity type."""
    matching_tools = []
    for server, tools in TOOL_DESCRIPTIONS.items():
        for tool_name, description in tools.items():
            if f"ENTITY TYPES:" in description:
                # Extract entity types from description
                entity_part = description.split("ENTITY TYPES:")[1].split(".")[0].lower()
                if entity_type.lower() in entity_part:
                    matching_tools.append({
                        "server": server,
                        "tool": tool_name,
                        "description": description
                    })
    return matching_tools


def get_tools_by_use_case(use_case: str) -> list:
    """Get all tools that match a specific use case."""
    matching_tools = []
    use_case_lower = use_case.lower()
    for server, tools in TOOL_DESCRIPTIONS.items():
        for tool_name, description in tools.items():
            if f"USE FOR:" in description:
                use_part = description.split("USE FOR:")[1].split(".")[0].lower()
                if use_case_lower in use_part:
                    matching_tools.append({
                        "server": server,
                        "tool": tool_name,
                        "description": description
                    })
    return matching_tools


def get_data_flow_info(server: str, tool: str) -> dict:
    """Extract data flow information (produces/requires) from tool description."""
    description = get_tool_description(server, tool)
    result = {"produces": [], "requires": []}

    if "DATA FLOW:" in description:
        data_flow = description.split("DATA FLOW:")[1].split(".")[0]
        if "Requires" in data_flow:
            requires_part = data_flow.split("Requires")[1].split(",")[0]
            result["requires"].append(requires_part.strip())
        if "produces" in data_flow.lower():
            produces_part = data_flow.lower().split("produces")[1]
            result["produces"].append(produces_part.strip())

    return result


# For testing
if __name__ == "__main__":
    # Test entity type lookup
    print("=== Tools for 'protein' entity ===")
    for tool in get_tools_by_entity_type("protein")[:5]:
        print(f"  {tool['server']}.{tool['tool']}")

    print("\n=== Tools for 'network analysis' use case ===")
    for tool in get_tools_by_use_case("network"):
        print(f"  {tool['server']}.{tool['tool']}")
