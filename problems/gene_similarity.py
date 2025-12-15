"""
Problem 1: Gene Function Similarity Analysis Handler
T 세포 유전자 간 기능 유사성 정량화
"""

from typing import Dict, Any, List
from .base_handler import BaseProblemHandler
from ..core import ResearchGoal, Hypothesis


class GeneSimilarityHandler(BaseProblemHandler):
    """Handler for T cell gene function similarity quantification"""
    
    def __init__(self):
        super().__init__("gene_function_similarity")
    
    def get_generation_prompt_additions(self, research_goal: ResearchGoal) -> str:
        return """
        ### Gene Function Similarity Analysis Context:
        - Focus on T cell gene regulatory networks
        - Consider expression correlation (expr_corr), structural similarity (struct_sim), 
          and phylogenetic similarity (phylo_sim)
        - Target identification of functionally similar gene pairs
        - Consider naive/memory T cells (resting) and activated T cells (TH0, TH1, TH2, TH17, TREG)
        
        ### Expected Analysis Components:
        1. LLM-based gene function summarization
        2. Knowledge-based phylogenetic tree construction
        3. Expression-based similarity scoring
        4. Integrated similarity score calculation
        5. Protein/domain structural similarity analysis
        
        ### Key Validation Criteria:
        - Biological relevance of gene pairs
        - Statistical significance of similarity scores
        - Consistency across multiple similarity metrics
        - Literature support for functional relationships
        """
    
    def get_review_criteria(self, research_goal: ResearchGoal) -> Dict[str, Any]:
        return {
            "required_components": [
                "expression_correlation_analysis",
                "structural_similarity_scoring",
                "phylogenetic_analysis",
                "integrated_similarity_metric"
            ],
            "validation_checks": [
                "statistical_significance",
                "biological_plausibility",
                "literature_consistency",
                "reproducibility"
            ],
            "output_requirements": [
                "gene_pair_rankings",
                "similarity_scores",
                "biological_functions",
                "supporting_evidence"
            ]
        }
    
    def validate_hypothesis(self, hypothesis: Hypothesis, research_goal: ResearchGoal) -> Dict[str, Any]:
        """Validate gene similarity hypothesis"""
        validation = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        content_lower = hypothesis.content.lower()
        
        # Check for key components
        required_terms = ["expression", "similarity", "gene"]
        missing_terms = [term for term in required_terms if term not in content_lower]
        
        if missing_terms:
            validation["warnings"].append(f"Missing key terms: {missing_terms}")
        
        # Check for quantitative analysis
        if not any(term in content_lower for term in ["score", "correlation", "metric"]):
            validation["warnings"].append("Lacks quantitative similarity metrics")
        
        # Check for biological context
        if not any(term in content_lower for term in ["t cell", "naive", "memory", "activated"]):
            validation["warnings"].append("Limited T cell context specificity")
        
        return validation
    
    def get_expected_outputs(self, research_goal: ResearchGoal) -> List[str]:
        return [
            "Gene pair similarity rankings with scores",
            "LLM-generated gene function summaries",
            "Phylogenetic tree (knowledge-based)",
            "Expression correlation matrix",
            "Protein structural similarity scores (top 300 genes)",
            "Key gene pairs for resting T cell regulation",
            "Biological function annotations",
            "Supporting literature references"
        ]
    
    def get_domain_knowledge(self) -> str:
        return """
        ### T Cell Biology Background:
        - T cells undergo activation from naive/memory states
        - Key transcription factors: TCF1, LEF1, FOXP1, RUNX3
        - Activation markers: CD69, CD25, CD44
        - Subsets: CD4+ (helper) and CD8+ (cytotoxic)
        - Differentiation pathways: TH1, TH2, TH17, TREG
        
        ### Relevant Gene Similarity Metrics:
        - Expression correlation: Pearson/Spearman correlation of TPM values
        - Structural similarity: Protein domain composition, 3D structure alignment
        - Phylogenetic similarity: Evolutionary conservation, ortholog relationships
        - Functional similarity: GO term overlap, pathway co-membership
        
        ### Data Integration Strategies:
        - Multi-modal integration of expression + structure + phylogeny
        - Weighted scoring schemes based on evidence strength
        - Machine learning for similarity prediction
        - Network-based community detection for gene clustering
        """
