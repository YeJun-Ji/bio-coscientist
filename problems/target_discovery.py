"""
Problem 4: Therapeutic Target Discovery Handler
폐 섬유증 치료 타겟 발굴: IL-11 상호작용 네트워크 기반 억제제 후보 설계
"""

from typing import Dict, Any, List
from .base_handler import BaseProblemHandler
from ..core import ResearchGoal, Hypothesis


class TargetDiscoveryHandler(BaseProblemHandler):
    """Handler for therapeutic target discovery (lung fibrosis / IL-11 network)"""
    
    def __init__(self):
        super().__init__("therapeutic_target_discovery")
    
    def get_generation_prompt_additions(self, research_goal: ResearchGoal) -> str:
        return """
        ### Target Discovery Context:
        - Disease: Idiopathic pulmonary fibrosis (IPF)
        - Key cytokine: IL-11 (Interleukin-11)
        - Pathological processes: Fibroblast activation, ECM accumulation, aging
        - Goal: Identify IL-11 network targets to block fibrosis pathway
        
        ### Analysis Requirements:
        
        (A) IL-11 and Interactor Analysis:
        1. Literature analysis (IL-11 in fibrosis and aging)
        2. IL-11 interactor identification and integration
        3. Protein-protein interaction (PPI) network modeling
        4. Fibrosis/aging pathway modeling
        5. Target candidate proposal with rationale
        
        (B) Binder/Inhibitor Design Strategy:
        1. Target class and modality definition
        2. Binder design strategy (antibody, small molecule, etc.)
        3. Hypotheses on fibrosis/aging inhibition and safety
        
        ### Key Scientific Questions:
        - What is IL-11's role in fibrosis and aging?
        - Who are the key IL-11 interactors?
        - Which network nodes are most impactful for intervention?
        - What therapeutic modality is most appropriate?
        - How to balance efficacy and safety?
        
        ### Data Sources to Consider:
        - PubMed: IL-11 fibrosis, IL-11 aging, IPF pathogenesis
        - STRING/BioGRID: PPI databases
        - OMIM/DisGeNET: Disease-gene associations
        - DrugBank/ChEMBL: Existing therapeutics
        - Tissue/cell-type expression databases
        """
    
    def get_review_criteria(self, research_goal: ResearchGoal) -> Dict[str, Any]:
        return {
            "required_components": [
                "il11_role_summary",
                "interactor_identification",
                "ppi_network_model",
                "pathway_analysis",
                "target_prioritization",
                "binder_strategy"
            ],
            "analysis_requirements": [
                "literature_integration",
                "network_analysis",
                "pathway_enrichment",
                "target_validation_strategy"
            ],
            "validation_checks": [
                "target_relevance_to_fibrosis",
                "druggability_assessment",
                "safety_considerations",
                "clinical_feasibility"
            ],
            "output_requirements": [
                "il11_function_summary",
                "interactor_list",
                "network_diagram",
                "target_rankings",
                "therapeutic_strategy",
                "efficacy_safety_hypothesis"
            ]
        }
    
    def validate_hypothesis(self, hypothesis: Hypothesis, research_goal: ResearchGoal) -> Dict[str, Any]:
        """Validate target discovery hypothesis"""
        validation = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        content_lower = hypothesis.content.lower()
        
        # Check for IL-11 focus
        if "il-11" not in content_lower and "interleukin-11" not in content_lower:
            validation["issues"].append("Missing IL-11 focus")
            validation["valid"] = False
        
        # Check for network/pathway analysis
        if not any(term in content_lower for term in ["network", "pathway", "interaction", "signaling"]):
            validation["warnings"].append("Lacks network/pathway analysis component")
        
        # Check for target identification
        if not any(term in content_lower for term in ["target", "inhibit", "block", "therapeutic"]):
            validation["warnings"].append("Should identify specific therapeutic targets")
        
        # Check for fibrosis context
        if not any(term in content_lower for term in ["fibrosis", "fibroblast", "ecm", "collagen"]):
            validation["warnings"].append("Limited fibrosis pathology context")
        
        # Check for therapeutic modality
        if not any(term in content_lower for term in ["antibody", "small molecule", "inhibitor", "binder"]):
            validation["warnings"].append("Should specify therapeutic modality")
        
        return validation
    
    def get_expected_outputs(self, research_goal: ResearchGoal) -> List[str]:
        return [
            "IL-11 role summary (fibrosis and aging)",
            "IL-11 interactor list with evidence",
            "PPI network model (nodes, edges, weights)",
            "Fibrosis/aging pathway diagram",
            "Top target candidates (ranked with rationale)",
            "Target class and modality recommendations",
            "Binder design strategy outline",
            "Efficacy and safety hypotheses",
            "Supporting literature references",
            "Alternative therapeutic approaches discussed"
        ]
    
    def get_domain_knowledge(self) -> str:
        return """
        ### Idiopathic Pulmonary Fibrosis (IPF):
        - Progressive lung disease with poor prognosis
        - Median survival: 3-5 years post-diagnosis
        - Current drugs: Pirfenidone, Nintedanib (slow progression only)
        - Pathology: Excessive ECM deposition, fibroblast activation
        - Key features: Aging-associated, irreversible scarring
        
        ### IL-11 Biology:
        - Cytokine of IL-6 family
        - Receptor: IL-11Rα + gp130 heterodimer
        - Signaling: JAK/STAT, MAPK, PI3K pathways
        - Functions: Hematopoiesis, inflammation, fibrosis
        - Recent findings: Pro-fibrotic and pro-aging factor
        
        ### Fibrosis Mechanisms:
        - TGF-β1 as master regulator
        - Myofibroblast differentiation and activation
        - ECM proteins: Collagen I, III, fibronectin
        - MMPs and TIMPs (matrix remodeling)
        - Epithelial-mesenchymal transition (EMT)
        - Senescence and SASP (senescence-associated secretory phenotype)
        
        ### Network Analysis Approaches:
        - PPI database integration (STRING, BioGRID, IntAct)
        - Pathway enrichment (KEGG, Reactome, GO)
        - Network centrality metrics (degree, betweenness, closeness)
        - Community detection algorithms
        - Disease gene prioritization (RWR, PRINCE, DIAMOnD)
        
        ### Therapeutic Modalities:
        - Monoclonal antibodies (anti-IL-11, anti-IL-11Rα)
        - Small molecule inhibitors (JAK inhibitors)
        - Antisense oligonucleotides (ASO)
        - siRNA/shRNA
        - Cell therapy (MSCs, engineered T cells)
        
        ### Drug Target Criteria:
        - Disease relevance (genetic/expression evidence)
        - Druggability (structure, binding sites)
        - Safety (tissue expression, knockouts)
        - Biomarkers for patient selection
        - Clinical precedent (related targets)
        """
