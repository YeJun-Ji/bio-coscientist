"""
Problem 2: RNA Stability Mechanism Investigation Handler
RNA 바이러스 유래 cis-Regulatory Element에 의한 mRNA 안정화 메커니즘 규명
"""

from typing import Dict, Any, List
from .base_handler import BaseProblemHandler
from ..core import ResearchGoal, Hypothesis


class RNAStabilityHandler(BaseProblemHandler):
    """Handler for RNA virus CRE-mediated mRNA stabilization mechanism"""
    
    def __init__(self):
        super().__init__("rna_stability_mechanism")
    
    def get_generation_prompt_additions(self, research_goal: ResearchGoal) -> str:
        return """
        ### RNA Stability Investigation Context:
        - Viral CRE (cis-regulatory element) inserted in 3' UTR
        - Nanopore direct RNA sequencing (dRNA-seq) data available
        - Active vs Inactive cell comparison (RNA stability phenotype)
        - Control vs CRE-containing RNA comparison
        - Poly(A) tail length and modification analysis required
        
        ### Data Files Context:
        - pod5 files: Raw electrical signal data from nanopore sequencing
        - BAM files: Basecalled and aligned reads
        - Active-control, active-cre, inactive-control, inactive-cre samples
        
        ### Expected Analysis Components:
        1. Molecular mechanism hypotheses (3+ mechanisms)
        2. Poly(A) tail length distribution analysis
        3. Modified base detection in poly(A) regions
        4. CRE-dependent modification pattern analysis
        5. Host enzyme identification (top 5 candidates)
        6. Experimental validation design
        7. Follow-up research directions
        8. Full manuscript draft (~3000 words, 10 figures)
        
        ### Key Scientific Questions:
        - How does CRE increase RNA stability?
        - What poly(A) modifications are present?
        - Which host enzymes mediate CRE function?
        - How do modifications differ between active/inactive states?
        """
    
    def get_review_criteria(self, research_goal: ResearchGoal) -> Dict[str, Any]:
        return {
            "required_components": [
                "molecular_mechanisms",
                "polya_tail_analysis",
                "modification_detection",
                "enzyme_candidates",
                "validation_strategy"
            ],
            "data_analysis_requirements": [
                "nanopore_signal_processing",
                "polya_length_quantification",
                "modified_base_calling",
                "statistical_comparison"
            ],
            "validation_checks": [
                "mechanism_plausibility",
                "data_interpretation_accuracy",
                "enzyme_candidate_rationale",
                "experimental_feasibility"
            ],
            "output_requirements": [
                "mechanism_hypotheses",
                "quantitative_polya_data",
                "modification_patterns",
                "enzyme_rankings",
                "validation_experiments",
                "manuscript_draft"
            ]
        }
    
    def validate_hypothesis(self, hypothesis: Hypothesis, research_goal: ResearchGoal) -> Dict[str, Any]:
        """Validate RNA stability mechanism hypothesis"""
        validation = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        content_lower = hypothesis.content.lower()
        
        # Check for mechanism description
        required_terms = ["cre", "poly(a)", "rna stability"]
        missing_terms = [term for term in required_terms if term not in content_lower]
        
        if missing_terms:
            validation["warnings"].append(f"Missing key concepts: {missing_terms}")
        
        # Check for data analysis component
        if not any(term in content_lower for term in ["nanopore", "drna-seq", "modification", "sequencing"]):
            validation["warnings"].append("Lacks specific data analysis strategy")
        
        # Check for enzyme involvement
        if not any(term in content_lower for term in ["enzyme", "protein", "pabp", "polymerase"]):
            validation["warnings"].append("Should identify host enzymes involved")
        
        # Check for experimental validation
        if not any(term in content_lower for term in ["experiment", "validation", "assay", "test"]):
            validation["warnings"].append("Missing experimental validation approach")
        
        return validation
    
    def get_expected_outputs(self, research_goal: ResearchGoal) -> List[str]:
        return [
            "3+ molecular mechanisms for CRE-mediated RNA stabilization",
            "Poly(A) tail length distributions (quantitative comparison)",
            "Modified base detection results with visualizations",
            "CRE-dependent modification patterns (active vs inactive)",
            "Top 5 host enzyme candidates with rankings and rationale",
            "Experimental validation strategy",
            "Follow-up research directions",
            "Manuscript draft (~3000 words, 10 figures)",
            "Statistical analyses and significance tests"
        ]
    
    def get_domain_knowledge(self) -> str:
        return """
        ### RNA Stability Mechanisms:
        - Poly(A) tail protection from 3'→5' exonucleases
        - Recruitment of poly(A) binding proteins (PABPs)
        - Circularization via PABP-eIF4G-eIF4E complex
        - Modified nucleotides (m6A, pseudouridine, etc.)
        - Secondary structure protection
        
        ### Relevant Enzymes:
        - PABPC1/PABPC4: Poly(A) binding proteins
        - TENT2/TENT4: Non-canonical poly(A) polymerases
        - METTL3/METTL14: m6A methyltransferases
        - YTHDF1/2/3: m6A readers
        - PUS1/7: Pseudouridine synthases
        - DIS3L2: 3'→5' exoribonuclease
        
        ### Nanopore dRNA-seq Analysis:
        - Pod5: Raw electrical current signals
        - Basecalling: Signal → nucleotide sequence
        - Poly(A) length: Extended homopolymer detection
        - Modified bases: Signal deviation from canonical bases
        - Tools: Nanopolish, Tombo, EpiNano, m6Anet
        
        ### Viral CRE Examples:
        - IRES elements (HCV, EMCV)
        - 3' UTR stability elements (alphaviruses)
        - Pseudoknots and stem-loops
        - Protein-binding motifs
        """
