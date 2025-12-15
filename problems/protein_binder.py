"""
Problem 3: Protein Binder Design Handler
삼중음성유방암(TNBC) 치료제 후보 설계: TNFα-ΔNp63α 축 표적 mini-binder
"""

from typing import Dict, Any, List
from .base_handler import BaseProblemHandler
from ..core import ResearchGoal, Hypothesis


class ProteinBinderHandler(BaseProblemHandler):
    """Handler for protein binder design (TNBC therapeutic mini-binder)"""
    
    def __init__(self):
        super().__init__("protein_binder_design")
    
    def get_generation_prompt_additions(self, research_goal: ResearchGoal) -> str:
        return """
        ### Protein Binder Design Context:
        - Target: TNFR1/2 (TNF receptors)
        - Therapeutic goal: TNBC treatment via TNFα-ΔNp63α axis modulation
        - Binder type: Mini-binder (compact protein scaffold)
        - Key considerations: Binding specificity, off-target minimization, immunogenicity
        - Pathway bias: TNFR1 vs TNFR2 selective activation/inhibition
        
        ### Required Pipeline Components:
        1. TNFR1/2 binding site selection
        2. Biophysical constraints (length, charge, stability, producibility)
        3. Candidate binder sequence design
        4. Binding structure and kinetics prediction (KD, kon, koff)
        5. Off-target receptor screening
        6. Candidate prioritization and ranking
        
        ### AI/ML Design Tools to Consider:
        - AlphaFold2/3 for structure prediction
        - RoseTTAFold for protein design
        - ProteinMPNN for sequence design
        - ESM, ProtGPT for sequence generation
        - FoldX, Rosetta for binding affinity prediction
        - NetMHCpan for immunogenicity screening
        
        ### Validation Criteria:
        - High binding affinity (KD < 10 nM)
        - TNFR pathway selectivity
        - Minimal off-target binding
        - Low immunogenicity risk
        - Biophysical feasibility
        - Developability (expression, stability, formulation)
        """
    
    def get_review_criteria(self, research_goal: ResearchGoal) -> Dict[str, Any]:
        return {
            "required_components": [
                "binding_site_analysis",
                "sequence_design_strategy",
                "structure_prediction",
                "affinity_prediction",
                "off_target_screening",
                "ranking_methodology"
            ],
            "design_constraints": [
                "length_constraint",
                "charge_distribution",
                "thermal_stability",
                "production_feasibility",
                "immunogenicity_assessment"
            ],
            "validation_checks": [
                "binding_affinity_target",
                "specificity_ratio",
                "structural_plausibility",
                "developability_score"
            ],
            "output_requirements": [
                "binder_sequences",
                "predicted_structures",
                "binding_parameters",
                "off_target_profiles",
                "ranked_candidates"
            ]
        }
    
    def validate_hypothesis(self, hypothesis: Hypothesis, research_goal: ResearchGoal) -> Dict[str, Any]:
        """Validate protein binder design hypothesis"""
        validation = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        content_lower = hypothesis.content.lower()
        
        # Check for target specification
        if not any(term in content_lower for term in ["tnfr", "tnf receptor"]):
            validation["warnings"].append("Missing specific target (TNFR1/2)")
        
        # Check for design strategy
        if not any(term in content_lower for term in ["sequence", "design", "scaffold", "binder"]):
            validation["issues"].append("Lacks clear binder design strategy")
            validation["valid"] = False
        
        # Check for binding affinity consideration
        if not any(term in content_lower for term in ["kd", "affinity", "binding", "interaction"]):
            validation["warnings"].append("Should include binding affinity predictions")
        
        # Check for specificity
        if not any(term in content_lower for term in ["specificity", "selectivity", "off-target"]):
            validation["warnings"].append("Missing specificity/off-target analysis")
        
        # Check for computational tools
        if not any(term in content_lower for term in ["alphafold", "rosetta", "design tool", "prediction"]):
            validation["warnings"].append("Should specify computational design tools")
        
        return validation
    
    def get_expected_outputs(self, research_goal: ResearchGoal) -> List[str]:
        return [
            "Selected TNFR1/2 binding sites with rationale",
            "Biophysical constraint specifications",
            "Candidate binder sequences (5-20 candidates)",
            "Predicted 3D structures (binder-TNFR complexes)",
            "Binding kinetics predictions (KD, kon, koff)",
            "Off-target receptor screening results",
            "Immunogenicity risk scores",
            "Ranked candidate list with scoring criteria",
            "Tool selection rationale for each pipeline step",
            "Alternative design approaches discussed"
        ]
    
    def get_domain_knowledge(self) -> str:
        return """
        ### TNBC and TNFα-ΔNp63α Axis:
        - TNBC: Triple-negative breast cancer (ER-, PR-, HER2-)
        - ΔNp63α: Key biomarker and therapeutic target in TNBC
        - TNFα signaling induces ΔNp63α degradation
        - TNFR1: Death receptor, pro-apoptotic
        - TNFR2: Pro-survival, immune modulation
        
        ### Mini-binder Design Strategies:
        - Scaffold options: Affibody, DARPin, Anticalin, Nanobody
        - Length: typically 40-150 amino acids
        - High stability, low immunogenicity
        - Easier production than full antibodies
        
        ### AI Tools for Protein Design:
        - AlphaFold2/3: Structure prediction (monomer/complex)
        - ESMFold: Fast structure prediction
        - ProteinMPNN: Sequence design from structure
        - RFdiffusion: Generative protein design
        - Rosetta: Detailed energy calculations, docking
        - FoldX: Stability and binding energy predictions
        
        ### Binding Affinity Targets:
        - Therapeutic antibodies: KD 0.1-10 nM
        - Mini-binders: KD 1-100 nM
        - Clinical candidates: typically <10 nM
        
        ### Off-Target Screening:
        - Sequence similarity search (BLAST)
        - Structural similarity (Dali, TM-align)
        - Binding prediction to related receptors
        - Cross-reactivity panel testing
        
        ### Immunogenicity Prediction:
        - MHC-II binding prediction (NetMHCIIpan)
        - T cell epitope prediction
        - Aggregation propensity
        - Post-translational modification sites
        """
