"""
Report Generator - 연구 결과를 읽기 쉬운 보고서로 변환
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


class ReportGenerator:
    """연구 결과를 Markdown 보고서로 변환"""
    
    def __init__(self, results_file: str):
        """
        Args:
            results_file: research_results.json 파일 경로
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.hypotheses = self.results.get('hypotheses', [])
        self.reviews = self.results.get('reviews', [])
        self.overviews = self.results.get('overviews', [])
    
    def generate_report(self, output_file: str = None) -> str:
        """완전한 연구 보고서 생성"""
        
        report_sections = [
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_methodology(),
            self._generate_hypothesis_results(),
            self._generate_review_analysis(),
            self._generate_evolution_analysis(),
            self._generate_recommendations(),
            self._generate_appendix()
        ]
        
        report = "\n\n".join(report_sections)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to: {output_file}")
        
        return report
    
    def _generate_header(self) -> str:
        """보고서 헤더"""
        return f"""# BioCoScientist Research Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**System:** BioCoScientist v1.0  
**Model:** Claude 4.5 Sonnet

---
"""
    
    def _generate_executive_summary(self) -> str:
        """요약"""
        total = len(self.hypotheses)
        
        # 상태별 카운트
        status_counts = {}
        for hyp in self.hypotheses:
            status = hyp.get('status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 카테고리별 카운트
        category_counts = {}
        for hyp in self.hypotheses:
            cat = hyp.get('category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Top 5 가설
        top_hypotheses = sorted(
            [h for h in self.hypotheses if h.get('status') not in ['HypothesisStatus.REJECTED']],
            key=lambda x: x.get('elo_rating', 1200),
            reverse=True
        )[:5]
        
        summary = f"""## Executive Summary

### Research Overview
- **Total Hypotheses Generated:** {total}
- **Research Iterations:** {len(self.overviews)}
- **Total Reviews Conducted:** {len(self.reviews)}

### Hypothesis Status Distribution
"""
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            summary += f"- **{status}:** {count} ({percentage:.1f}%)\n"
        
        summary += f"\n### Category Distribution\n"
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            summary += f"- **{cat}:** {count} ({percentage:.1f}%)\n"
        
        if top_hypotheses:
            summary += f"\n### Top {len(top_hypotheses)} Hypotheses\n"
            for i, hyp in enumerate(top_hypotheses, 1):
                summary += f"\n#### {i}. {hyp.get('id', 'Unknown')}\n"
                summary += f"**Category:** {hyp.get('category', 'N/A')}  \n"
                summary += f"**Status:** {hyp.get('status', 'N/A')}  \n"
                summary += f"**ELO Rating:** {hyp.get('elo_rating', 1200):.0f}  \n"
                summary += f"**Summary:** {hyp.get('summary', 'N/A')}\n"
        
        return summary
    
    def _generate_methodology(self) -> str:
        """방법론 섹션"""
        
        # 생성 방법 분석
        generation_methods = {}
        for hyp in self.hypotheses:
            hyp_id = hyp.get('id', '')
            if 'lit_' in hyp_id:
                method = 'Literature-based'
            elif 'debate_' in hyp_id:
                method = 'Debate-based'
            elif 'assume_' in hyp_id:
                method = 'Assumption-based'
            elif 'expand_' in hyp_id:
                method = 'Research Space Expansion'
            elif 'coherent_' in hyp_id:
                method = 'Coherence Improvement'
            elif 'combined_' in hyp_id:
                method = 'Hypothesis Combination'
            elif 'simple_' in hyp_id:
                method = 'Simplification'
            elif 'divergent_' in hyp_id:
                method = 'Divergent Generation'
            elif 'ground_' in hyp_id:
                method = 'Literature Grounding'
            else:
                method = 'Other'
            
            generation_methods[method] = generation_methods.get(method, 0) + 1
        
        methodology = """## Methodology

### Hypothesis Generation Methods
The system employed multiple strategies to generate diverse research hypotheses:

"""
        for method, count in sorted(generation_methods.items(), key=lambda x: x[1], reverse=True):
            methodology += f"- **{method}:** {count} hypotheses\n"
        
        methodology += """
### Review Process
Hypotheses underwent multi-stage evaluation:

1. **Initial Review** - Fast screening for basic correctness and quality
2. **Full Review** - Comprehensive evaluation with literature context
3. **Deep Verification** - Rigorous assumption-by-assumption validation
4. **Tournament Ranking** - Comparative evaluation through debates

### Scoring Metrics
- **Correctness:** Scientific accuracy and validity
- **Quality:** Clarity, specificity, and experimental feasibility
- **Novelty:** Originality and innovation
- **ELO Rating:** Competitive ranking through tournament matches
"""
        return methodology
    
    def _generate_hypothesis_results(self) -> str:
        """가설 결과 상세"""
        
        # 통과한 가설들
        passed = [h for h in self.hypotheses if 'REJECTED' not in h.get('status', '')]
        
        results = """## Detailed Hypothesis Results

"""
        
        if passed:
            results += f"### ✅ Accepted Hypotheses ({len(passed)})\n\n"
            
            for i, hyp in enumerate(sorted(passed, key=lambda x: x.get('elo_rating', 1200), reverse=True), 1):
                results += f"#### {i}. {hyp.get('id', 'Unknown')}\n\n"
                results += f"**Category:** {hyp.get('category', 'N/A')}  \n"
                results += f"**Status:** {hyp.get('status', 'N/A')}  \n"
                results += f"**Generated:** {hyp.get('generated_at', 'N/A')}  \n"
                results += f"**ELO Rating:** {hyp.get('elo_rating', 1200):.0f}  \n\n"
                
                results += f"**Summary:**  \n{hyp.get('summary', 'N/A')}\n\n"
                
                results += f"**Full Content:**  \n{hyp.get('content', 'N/A')}\n\n"
                
                # 리뷰 정보
                if hyp.get('initial_review'):
                    review = hyp['initial_review']
                    results += f"**Initial Review:**  \n"
                    results += f"- Correctness: {review.get('correctness_score', 'N/A')}/10  \n"
                    results += f"- Quality: {review.get('quality_score', 'N/A')}/10  \n"
                    results += f"- Novelty: {review.get('novelty_score', 'N/A')}/10  \n\n"
                
                # 진화 정보
                if hyp.get('parent_ids'):
                    results += f"**Evolved from:** {', '.join(hyp['parent_ids'])}  \n"
                    # Get evolution_method from metadata if not in top level
                    evolution_method = hyp.get('evolution_method') or hyp.get('metadata', {}).get('evolution_method', 'N/A')
                    results += f"**Evolution method:** {evolution_method}  \n\n"
                
                results += "---\n\n"
        
        else:
            results += "⚠️ No hypotheses passed the review process.\n\n"
        
        # 거부된 가설 샘플
        rejected = [h for h in self.hypotheses if 'REJECTED' in h.get('status', '')]
        if rejected:
            results += f"\n### ❌ Rejected Hypotheses ({len(rejected)})\n\n"
            results += f"Showing top 3 rejected hypotheses by initial scores:\n\n"
            
            # 초기 점수로 정렬
            top_rejected = sorted(
                rejected,
                key=lambda x: (
                    x.get('initial_review', {}).get('correctness_score', 0) +
                    x.get('initial_review', {}).get('quality_score', 0) +
                    x.get('initial_review', {}).get('novelty_score', 0)
                ),
                reverse=True
            )[:3]
            
            for i, hyp in enumerate(top_rejected, 1):
                results += f"#### {i}. {hyp.get('id', 'Unknown')}\n\n"
                results += f"**Summary:** {hyp.get('summary', 'N/A')}\n\n"
                
                # 거부 이유
                if hyp.get('initial_review'):
                    review = hyp['initial_review']
                    if not review.get('pass_review', False):
                        results += f"**Rejection Reason:** {review.get('summary', 'N/A')}\n\n"
                
                results += "---\n\n"
        
        return results
    
    def _generate_review_analysis(self) -> str:
        """리뷰 분석"""
        
        analysis = """## Review Analysis

### Review Statistics
"""
        
        # 리뷰 타입별 통계
        review_types = {}
        for review in self.reviews:
            rtype = review.get('review_type', 'unknown')
            review_types[rtype] = review_types.get(rtype, 0) + 1
        
        analysis += "#### Reviews by Type\n"
        for rtype, count in sorted(review_types.items(), key=lambda x: x[1], reverse=True):
            analysis += f"- **{rtype}:** {count}\n"
        
        # 통과율
        initial_reviews = [r for r in self.reviews if r.get('review_type') == 'initial']
        if initial_reviews:
            passed = sum(1 for r in initial_reviews if r.get('pass_review', False))
            pass_rate = (passed / len(initial_reviews) * 100) if initial_reviews else 0
            analysis += f"\n#### Initial Review Pass Rate\n"
            analysis += f"- **Passed:** {passed}/{len(initial_reviews)} ({pass_rate:.1f}%)\n"
        
        # 평균 점수
        if initial_reviews:
            avg_correctness = sum(r.get('correctness_score', 0) for r in initial_reviews) / len(initial_reviews)
            avg_quality = sum(r.get('quality_score', 0) for r in initial_reviews) / len(initial_reviews)
            avg_novelty = sum(r.get('novelty_score', 0) for r in initial_reviews) / len(initial_reviews)
            
            analysis += f"\n#### Average Scores (Initial Reviews)\n"
            analysis += f"- **Correctness:** {avg_correctness:.2f}/10\n"
            analysis += f"- **Quality:** {avg_quality:.2f}/10\n"
            analysis += f"- **Novelty:** {avg_novelty:.2f}/10\n"
        
        return analysis
    
    def _generate_evolution_analysis(self) -> str:
        """진화 분석"""
        
        evolved = [h for h in self.hypotheses if h.get('parent_ids')]
        
        analysis = f"""## Hypothesis Evolution Analysis

### Evolution Statistics
- **Total Evolved Hypotheses:** {len(evolved)}
- **Base Hypotheses:** {len(self.hypotheses) - len(evolved)}

"""
        
        if evolved:
            # 진화 방법별 통계
            evolution_methods = {}
            for hyp in evolved:
                method = hyp.get('evolution_method', 'unknown')
                evolution_methods[method] = evolution_methods.get(method, 0) + 1
            
            analysis += "#### Evolution Methods Used\n"
            for method, count in sorted(evolution_methods.items(), key=lambda x: x[1], reverse=True):
                analysis += f"- **{method}:** {count}\n"
            
            # 진화 성공률
            evolved_passed = [h for h in evolved if 'REJECTED' not in h.get('status', '')]
            success_rate = (len(evolved_passed) / len(evolved) * 100) if evolved else 0
            
            analysis += f"\n#### Evolution Success Rate\n"
            analysis += f"- **Passed:** {len(evolved_passed)}/{len(evolved)} ({success_rate:.1f}%)\n"
        
        return analysis
    
    def _generate_recommendations(self) -> str:
        """개선 권장사항"""
        
        return """## Recommendations

### For Future Research
1. **Hypothesis Refinement**
   - Focus on hypotheses with higher initial review scores
   - Balance specificity with verifiability
   - Strengthen connections to existing literature

2. **Review Process**
   - Consider adjusting deep verification thresholds
   - Implement staged validation for complex hypotheses
   - Add partial credit for partially validated claims

3. **Search Strategy**
   - Use more specific, targeted keywords
   - Expand search to include related concepts
   - Consider domain-specific databases

4. **Evolution Strategy**
   - Prioritize evolution of high-scoring base hypotheses
   - Combine successful elements from multiple approaches
   - Iterate on promising directions more aggressively

### System Improvements
- ✅ Enhanced logging with purpose tracking
- ✅ Automated report generation
- 🔄 Consider implementing adaptive review thresholds
- 🔄 Add visualization for hypothesis relationship graphs
"""
    
    def _generate_appendix(self) -> str:
        """부록"""
        
        return f"""## Appendix

### Data Files
- **JSON Results:** `research_results.json`
- **Full Hypothesis Data:** {len(self.hypotheses)} entries
- **Review Data:** {len(self.reviews)} entries
- **Overview Data:** {len(self.overviews)} entries

### Glossary
- **Initial Review:** Quick screening for basic quality
- **Full Review:** Comprehensive evaluation with literature
- **Deep Verification:** Rigorous claim-by-claim validation
- **ELO Rating:** Competitive ranking score (default: 1200)
- **Evolution:** Systematic improvement of existing hypotheses

### System Configuration
- **LLM Model:** Claude 4.5 Sonnet (via OpenRouter)
- **Temperature Range:** 0.2 - 0.9 (purpose-dependent)
- **Max Tokens:** 8192
- **Search Sources:** PubMed, Semantic Scholar

---

*Report generated by BioCoScientist Report Generator*
"""
    
    def generate_summary_report(self, output_file: str = None) -> str:
        """간단한 요약 보고서만 생성"""
        
        report = "\n\n".join([
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_recommendations()
        ])
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Summary report saved to: {output_file}")
        
        return report


def generate_report_from_json(
    json_file: str,
    output_file: str = None,
    report_type: str = "full"
) -> str:
    """
    편의 함수: JSON 파일에서 바로 보고서 생성
    
    Args:
        json_file: research_results.json 파일 경로
        output_file: 출력 파일 경로 (None이면 반환만)
        report_type: "full" 또는 "summary"
    
    Returns:
        생성된 보고서 텍스트
    """
    generator = ReportGenerator(json_file)
    
    if report_type == "summary":
        return generator.generate_summary_report(output_file)
    else:
        return generator.generate_report(output_file)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <research_results.json> [output.txt] [full|summary]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    report_type = sys.argv[3] if len(sys.argv) > 3 else "full"
    
    if not output_file:
        # 기본 출력 파일명 생성
        base_name = Path(json_file).stem
        output_file = f"{base_name}_report.txt"
    
    report = generate_report_from_json(json_file, output_file, report_type)
    print(f"\n✅ Report generation complete!")
    print(f"📄 Output: {output_file}")
