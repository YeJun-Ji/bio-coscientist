"""
Problem 5: Drug Repositioning Handler
T 세포 탈진 억제 약물 재활용(drug repositioning) 후보 예측
"""

from typing import Dict, Any, List
from .base_handler import BaseProblemHandler
from ..core import ResearchGoal, Hypothesis


class DrugRepositioningHandler(BaseProblemHandler):
    """Handler for drug repositioning (T cell exhaustion reversal)"""
    
    def __init__(self):
        super().__init__("drug_repositioning")
    
    def get_generation_prompt_additions(self, research_goal: ResearchGoal) -> str:
        return """
        ### Drug Repositioning Context:
        - Target condition: T cell exhaustion in chronic infection and cancer
        - Goal: Identify existing drugs that reverse exhaustion signature
        - Approach: Signature reversal using drug-gene databases
        - Data: Exhaustion gene signature + drug-target-transcriptome databases
        
        ### Analysis Requirements:
        
        (A) T Cell Exhaustion Signature Analysis:
        - Upregulated genes: PD-1, LAG-3, TIM-3, CTLA-4, TOX, etc.
        - Downregulated genes: IL-2, IFNγ, effector functions
        - Transcriptomic characterization
        - Pathway enrichment
        
        (B) Drug-Gene Network Analysis:
        - Drug-target protein/gene connections
        - Approved/investigational drug databases
        - Gene expression modulation by drugs
        - Connectivity Map (CMap) approach
        
        (C) Drug Candidate Prioritization:
        - Signature reversal scoring
        - Mechanism of action hypothesis
        - Clinical feasibility
        - Combination potential with checkpoint inhibitors
        
        ### Key Scientific Questions:
        - What genes define T cell exhaustion signature?
        - Which drugs can reverse this signature?
        - What are the mechanisms of reversal?
        - Which candidates have best clinical potential?
        - Can drugs synergize with existing immunotherapies?
        
        ### Data Sources to Utilize:
        - Drug databases: DrugBank, ChEMBL, PubChem, Pharos
        - Gene expression: GEO, TCGA, ImmGen
        - Drug-target: STITCH, BindingDB
        - Perturbation databases: LINCS L1000, CMap
        - Clinical trials: ClinicalTrials.gov
        """
    
    def get_review_criteria(self, research_goal: ResearchGoal) -> Dict[str, Any]:
        return {
            "required_components": [
                "exhaustion_signature_definition",
                "drug_target_network",
                "signature_reversal_analysis",
                "candidate_prioritization",
                "mechanism_hypotheses"
            ],
            "analysis_requirements": [
                "signature_scoring_method",
                "statistical_significance",
                "literature_validation",
                "clinical_feasibility_check"
            ],
            "validation_checks": [
                "signature_accuracy",
                "drug_target_evidence",
                "reversal_mechanism_plausibility",
                "safety_profile",
                "combination_potential"
            ],
            "output_requirements": [
                "exhaustion_gene_list",
                "drug_candidate_rankings",
                "reversal_scores",
                "mechanism_of_action",
                "clinical_recommendations"
            ]
        }
    
    def validate_hypothesis(self, hypothesis: Hypothesis, research_goal: ResearchGoal) -> Dict[str, Any]:
        """Validate drug repositioning hypothesis"""
        validation = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        content_lower = hypothesis.content.lower()
        
        # Check for exhaustion context
        if not any(term in content_lower for term in ["exhaustion", "exhausted", "pd-1", "lag-3", "tim-3"]):
            validation["issues"].append("Missing T cell exhaustion context")
            validation["valid"] = False
        
        # Check for drug identification
        if not any(term in content_lower for term in ["drug", "compound", "therapeutic", "medication"]):
            validation["issues"].append("No drug candidates identified")
            validation["valid"] = False
        
        # Check for mechanism
        if not any(term in content_lower for term in ["mechanism", "pathway", "target", "modulate"]):
            validation["warnings"].append("Should explain mechanism of action")
        
        # Check for signature reversal concept
        if not any(term in content_lower for term in ["signature", "expression", "reversal", "reverse"]):
            validation["warnings"].append("Lacks signature reversal analysis")
        
        # Check for clinical context
        if not any(term in content_lower for term in ["clinical", "trial", "therapy", "treatment"]):
            validation["warnings"].append("Should consider clinical feasibility")
        
        return validation
    
    def get_expected_outputs(self, research_goal: ResearchGoal) -> List[str]:
        return [
            "T cell exhaustion gene signature (up/down regulated)",
            "Signature enrichment analysis (pathways, GO terms)",
            "Drug-target network diagram",
            "Top drug candidates (ranked by reversal score)",
            "Mechanism of action hypotheses for each candidate",
            "Statistical significance assessments",
            "Literature support for candidates",
            "Clinical trial status and feasibility",
            "Combination therapy potential (with PD-1/CTLA-4 blockade)",
            "Safety and toxicity considerations"
        ]
    
    def get_domain_knowledge(self) -> str:
        return """
        ### T Cell Exhaustion Biology:
        - Phenotype: Functional impairment from chronic antigen stimulation
        - Context: Chronic infections (HIV, HCV, HBV), cancer (TME)
        - Markers: PD-1, LAG-3, TIM-3, CTLA-4, TIGIT, 2B4, CD160
        - Transcription: TOX, NR4A family, NFAT, Eomes, T-bet
        - Lost functions: Proliferation, cytokine production, cytotoxicity
        - Metabolic defects: Impaired glycolysis, mitochondrial dysfunction
        
        ### Exhaustion Reversal Strategies:
        - Checkpoint blockade: Anti-PD-1, anti-CTLA-4, anti-LAG-3
        - Metabolic reprogramming: Glucose, glutamine, fatty acid pathways
        - Epigenetic modulation: HDAC inhibitors, DNA methylation
        - Co-stimulation: 4-1BB, OX40, CD28 agonists
        - Cytokine support: IL-2, IL-7, IL-15, IL-21
        
        ### Drug Repositioning Approaches:
        - Signature reversal: Connectivity Map (CMap), LINCS L1000
        - Network-based: Drug-target-disease networks
        - Structure-based: Target structure similarity
        - Phenotypic screening: Direct functional assays
        
        ### Relevant Drug Classes:
        - Immunomodulators: Thalidomide analogs, interferons
        - Metabolic drugs: Metformin, statins, PPAR agonists
        - Epigenetic drugs: Azacitidine, vorinostat, romidepsin
        - Kinase inhibitors: Ibrutinib, dasatinib, ruxolitinib
        - Natural products: Curcumin, resveratrol, green tea extracts
        
        ### Database Resources:
        - DrugBank: Comprehensive drug data
        - LINCS L1000: Gene expression signatures of drug perturbations
        - ChEMBL: Bioactivity database
        - Pharos: Illuminating the Druggable Genome
        - ClinicalTrials.gov: Clinical trial registry
        - BindingDB: Drug-target binding affinities
        
        ### Signature Reversal Metrics:
        - Connectivity score: Similarity/anti-similarity to query signature
        - Enrichment score: GSEA-like approach
        - Kolmogorov-Smirnov statistic
        - Rank-based metrics
        - Bayesian approaches
        
        ### Clinical Translation Considerations:
        - Existing safety data (FDA-approved drugs preferred)
        - Blood-brain barrier penetration (if CNS involvement)
        - Drug-drug interactions (with checkpoint inhibitors)
        - Dosing and formulation
        - Biomarkers for patient selection
        - Combination synergy potential
        """
