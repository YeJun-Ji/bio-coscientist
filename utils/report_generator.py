"""
Report Generator - 연구 결과를 읽기 쉬운 보고서로 변환
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..external_apis import LLMClient
from ..prompts import PromptManager


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


class FinalResearchReportGenerator:
    """
    Generate comprehensive research report focused on the best hypothesis.

    This is the new report format that emphasizes actionable research output
    rather than just statistics about the research process.
    """

    def __init__(
        self,
        results: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Args:
            results: Results dict from supervisor_agent._finalize_research()
            llm_client: Optional LLM client for synthesizing step answers
            prompt_manager: Optional prompt manager for templates
        """
        self.results = results or {}

        # Safe extraction with None checks - if value is None, use default
        self.research_goal = results.get("research_goal") or {}
        self.best_hypothesis = results.get("best_hypothesis") or {}
        self.best_hypothesis_reviews = results.get("best_hypothesis_reviews") or []
        self.best_hypothesis_evolution = results.get("best_hypothesis_evolution") or []
        self.meta_reviews = results.get("meta_reviews") or []
        self.latest_meta_review = results.get("latest_meta_review") or {}
        self.execution_stats = results.get("execution_stats") or {}
        self.top_hypotheses = results.get("top_hypotheses") or []

        # Step-based parsing (Problem-Agnostic)
        self.parsed_problem = results.get("parsed_problem") or {}
        self.research_steps = results.get("research_steps") or []

        # LLM for synthesizing step answers into natural text
        self.llm = llm_client
        self.prompt_manager = prompt_manager or PromptManager()

    def generate(self, output_file: str = None) -> str:
        """Generate the complete research report"""
        sections = [
            self._generate_header(),
            self._generate_executive_summary(),
        ]

        # Add step-based sections if research_steps available
        if self.research_steps:
            sections.append(self._generate_problem_steps_overview())
            sections.append(self._generate_step_by_step_answers())
        else:
            sections.append(self._generate_recommended_hypothesis())

        sections.extend([
            self._generate_evidence_validation(),
            self._generate_research_insights(),
            self._generate_future_directions(),
            self._generate_methodology_summary()
        ])

        report = "\n\n".join(sections)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)

        return report

    def _generate_header(self) -> str:
        """Generate report header with dynamic title based on research goal"""
        goal_desc = self.research_goal.get("description", "Research")
        domain = self.research_goal.get("domain", "Biomedical")

        # Create dynamic title from research goal
        if len(goal_desc) > 80:
            title = goal_desc[:77] + "..."
        else:
            title = goal_desc

        return f"""# Research Report: {title}

**Domain:** {domain}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**System:** BioCoScientist v3.0

---"""

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section"""
        # Research goal summary
        goal_desc = self.research_goal.get("description", "N/A")

        # Best hypothesis summary
        best_hyp = self.best_hypothesis
        best_summary = best_hyp.get("summary", best_hyp.get("content", "N/A")[:200])

        # Key findings from meta review
        key_findings = []
        if self.latest_meta_review:
            patterns = self.latest_meta_review.get("key_patterns", [])
            if patterns:
                key_findings = patterns[:5]

        # If no meta review patterns, extract from hypothesis
        if not key_findings and best_hyp:
            content = best_hyp.get("content", "")
            # Extract key points from hypothesis
            key_findings = ["Detailed hypothesis content available in Section 2"]

        summary = f"""## 1. Executive Summary

### Research Objective
{goal_desc}

### Final Recommendation
{best_summary}

### Key Findings"""

        if key_findings:
            for finding in key_findings:
                summary += f"\n- {finding}"
        else:
            summary += "\n- See detailed hypothesis section below"

        return summary

    def _generate_recommended_hypothesis(self) -> str:
        """Generate detailed section for the best hypothesis"""
        hyp = self.best_hypothesis

        if not hyp:
            return """## 2. Recommended Hypothesis

⚠️ No hypothesis data available."""

        # Basic info
        hyp_id = hyp.get("id", "N/A")
        category = hyp.get("category", "N/A")
        elo_rating = hyp.get("elo_rating", 1200.0)
        novelty_score = hyp.get("novelty_score", 0)
        feasibility_score = hyp.get("feasibility_score", 0)
        wins = hyp.get("wins", 0)
        losses = hyp.get("losses", 0)

        section = f"""## 2. Recommended Hypothesis

### 2.1 Overview

| Metric | Value |
|--------|-------|
| **ID** | {hyp_id} |
| **Category** | {category} |
| **ELO Rating** | {elo_rating:.0f} |
| **Novelty Score** | {novelty_score:.2f}/10 |
| **Feasibility Score** | {feasibility_score:.2f}/10 |
| **Tournament Record** | {wins}W - {losses}L |

### 2.2 Hypothesis Content

{hyp.get("content", "N/A")}

### 2.3 Scientific Rationale"""

        # Add rationale from reviews if available
        rationale_points = []
        for review in self.best_hypothesis_reviews:
            if isinstance(review, dict):
                quality_assess = review.get("quality_assessment", {})
                novelty_assess = review.get("novelty_assessment", {})

                if isinstance(quality_assess, dict) and quality_assess.get("reasoning"):
                    rationale_points.append(f"Quality: {quality_assess['reasoning']}")
                if isinstance(novelty_assess, dict) and novelty_assess.get("reasoning"):
                    rationale_points.append(f"Novelty: {novelty_assess['reasoning']}")

        if rationale_points:
            for point in rationale_points[:3]:
                section += f"\n- {point}"
        else:
            section += "\n- Based on systematic hypothesis generation and tournament ranking"
            section += "\n- Validated through multi-stage review process"

        # Proposed constraints if available
        constraints = hyp.get("proposed_constraints", {})
        if constraints:
            section += "\n\n### 2.4 Proposed Constraints\n"
            for key, value in constraints.items():
                section += f"\n- **{key.replace('_', ' ').title()}:** {value}"

        return section

    def _generate_problem_steps_overview(self) -> str:
        """Generate overview of research steps parsed from the problem"""
        if not self.research_steps:
            return ""

        parsed = self.parsed_problem
        problem_title = parsed.get("title", "Research Problem")
        problem_type = parsed.get("problem_type", "flat")
        format_detected = parsed.get("format_detected", "unknown")
        background = parsed.get("background", "")

        section = f"""## 2. Research Problem Structure

### 2.1 Problem Overview

**Title:** {problem_title}
**Type:** {problem_type.capitalize()}
**Format:** {format_detected}

### 2.2 Background

{background[:1000]}{"..." if len(background) > 1000 else ""}

### 2.3 Research Steps Overview

| Step | Title | Type | Dependencies |
|------|-------|------|--------------|"""

        for step in self.research_steps:
            step_id = step.get("step_id", "")
            title = step.get("title", "")[:40]
            step_type = step.get("step_type", "analysis")
            depends = ", ".join(step.get("depends_on", [])) or "-"
            section += f"\n| {step_id} | {title} | {step_type} | {depends} |"

        # Add input data if available
        input_data = parsed.get("input_data_description")
        if input_data:
            section += f"\n\n### 2.4 Input Data\n\n{input_data}"

        return section

    def _generate_step_by_step_answers(self) -> str:
        """Generate step-by-step answers with LLM-synthesized natural text."""
        if not self.research_steps:
            return ""

        lines = []

        # 1. Section header
        lines.append("## Research Problem Answers\n")

        hyp = self.best_hypothesis
        step_answers = hyp.get("step_answers", {}) if hyp else {}

        # 2. Brief completion summary
        answered = len(step_answers)
        total = len(self.research_steps)
        if total > 0:
            lines.append(f"> {answered} of {total} research steps answered ({answered/total*100:.1f}%)\n")

        # 3. Per-step answers (synthesized or fallback)
        for step in self.research_steps:
            step_id = step.get("step_id", "")
            step_title = step.get("title", "")

            lines.append(f"### {step_id}. {step_title}\n")

            answer_data = step_answers.get(step_id, {})

            if answer_data:
                answer = answer_data.get("answer", "")
                rationale = answer_data.get("rationale", "")
                deliverables = answer_data.get("deliverables", {})
                evidence = answer_data.get("evidence", [])
                confidence = answer_data.get("confidence", 0)

                # Try LLM synthesis if available
                if self.llm and answer and rationale:
                    synthesized = self._synthesize_step_answer(
                        step_id=step_id,
                        step_title=step_title,
                        answer=answer,
                        rationale=rationale,
                        deliverables=deliverables
                    )
                    if synthesized:
                        lines.append(f"{synthesized}\n")
                    else:
                        # Fallback if synthesis fails
                        lines.append(f"{answer}\n")
                        if rationale:
                            lines.append(f"{rationale}\n")
                else:
                    # No LLM: use simple format
                    if answer:
                        lines.append(f"{answer}\n")
                    if rationale:
                        lines.append(f"{rationale}\n")

                # References (always show separately)
                if evidence:
                    evidence_str = ", ".join(str(e) for e in evidence[:5])
                    lines.append(f"**References:** {evidence_str}\n")

                # Confidence (always show separately)
                lines.append(f"**Confidence:** {confidence*100:.1f}%\n")
            else:
                lines.append("*⚠️ No answer generated for this step.*\n")

            lines.append("---\n")

        # No quality evaluation section (user preference: clean output)

        return "\n".join(lines)

    def _synthesize_step_answer(
        self,
        step_id: str,
        step_title: str,
        answer: str,
        rationale: str,
        deliverables: Dict[str, Any]
    ) -> Optional[str]:
        """
        Synthesize Answer, Rationale, and Deliverables into a natural flowing paragraph using LLM.

        Args:
            step_id: Step identifier
            step_title: Step title
            answer: The answer content
            rationale: The reasoning/rationale
            deliverables: Dict of deliverables produced

        Returns:
            Synthesized paragraph or None if synthesis fails
        """
        if not self.llm:
            return None

        try:
            # Build prompt using template
            prompt = self.prompt_manager.get_prompt(
                "report/synthesize_step_answer",
                step_id=step_id,
                step_title=step_title,
                answer=answer,
                rationale=rationale,
                deliverables=deliverables
            )

            # Call LLM synchronously (wrap async in sync for report generation)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.llm.generate(
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=500,
                            purpose="synthesize_step_answer"
                        )
                    )
                    synthesized = future.result(timeout=30)
            else:
                synthesized = loop.run_until_complete(
                    self.llm.generate(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=500,
                        purpose="synthesize_step_answer"
                    )
                )

            return synthesized.strip() if synthesized else None

        except Exception as e:
            # Log error and return None to trigger fallback
            print(f"Warning: Step answer synthesis failed for step {step_id}: {e}")
            return None

    def _generate_evidence_validation(self) -> str:
        """Generate evidence and validation section"""
        section = """## 3. Evidence & Validation

### 3.1 Review Summary"""

        # Summarize reviews
        reviews = self.best_hypothesis_reviews
        if reviews:
            for i, review in enumerate(reviews, 1):
                if isinstance(review, dict):
                    review_type = review.get("review_type", "unknown")
                    passed = review.get("pass_review", False)
                    status = "✅ PASSED" if passed else "❌ FAILED"

                    section += f"\n\n**Review {i} ({review_type}):** {status}"

                    # Add concerns if any
                    concerns = review.get("concerns", [])
                    if concerns:
                        section += "\n- Concerns raised:"
                        for concern in concerns[:3]:
                            section += f"\n  - {concern}"

                    # Add suggestions
                    suggestions = review.get("suggestions", [])
                    if suggestions:
                        section += "\n- Suggestions:"
                        for suggestion in suggestions[:3]:
                            section += f"\n  - {suggestion}"
        else:
            section += "\n\n_No review data available_"

        # Evolution lineage
        section += "\n\n### 3.2 Evolution Lineage"

        evolution = self.best_hypothesis_evolution
        if evolution:
            section += "\n\nThis hypothesis evolved through the following stages:\n"
            for i, ancestor in enumerate(evolution):
                if isinstance(ancestor, dict):
                    anc_id = ancestor.get("id", "unknown")
                    anc_summary = ancestor.get("summary", "")[:100]
                    section += f"\n{i+1}. **{anc_id}**: {anc_summary}..."
        else:
            section += "\n\n_This is a first-generation hypothesis (no evolution history)_"

        # Tournament performance
        section += "\n\n### 3.3 Tournament Performance"

        hyp = self.best_hypothesis
        if hyp:
            wins = hyp.get("wins", 0)
            losses = hyp.get("losses", 0)
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0

            section += f"""

| Statistic | Value |
|-----------|-------|
| Total Matches | {total} |
| Wins | {wins} |
| Losses | {losses} |
| Win Rate | {win_rate:.1f}% |"""

        return section

    def _generate_research_insights(self) -> str:
        """Generate research insights from meta-review"""
        section = """## 4. Research Insights

### 4.1 Key Patterns Discovered"""

        if self.latest_meta_review:
            patterns = self.latest_meta_review.get("key_patterns", [])
            if patterns:
                for pattern in patterns:
                    section += f"\n- {pattern}"
            else:
                section += "\n\n_No patterns identified in meta-review_"

            # Add trends
            trends = self.latest_meta_review.get("emerging_trends", [])
            if trends:
                section += "\n\n### 4.2 Emerging Trends\n"
                for trend in trends:
                    section += f"\n- {trend}"
        else:
            section += "\n\n_Meta-review data not available_"

        # Decision factors from tournament
        section += "\n\n### 4.3 Decision Factors"
        section += "\n\nThe recommended hypothesis was selected based on:"
        section += "\n- ELO tournament ranking (pairwise comparisons)"
        section += "\n- Multi-stage review process"
        section += "\n- Scientific validity and novelty assessment"

        return section

    def _generate_future_directions(self) -> str:
        """Generate future directions section"""
        section = """## 5. Future Directions

### 5.1 Recommended Next Steps"""

        # Add suggestions from meta-review if available
        if self.latest_meta_review:
            suggestions = self.latest_meta_review.get("research_suggestions", [])
            unexplored = self.latest_meta_review.get("unexplored_areas", [])

            if suggestions:
                for suggestion in suggestions[:5]:
                    section += f"\n1. {suggestion}"

            if unexplored:
                section += "\n\n### 5.2 Unexplored Areas\n"
                for area in unexplored[:5]:
                    section += f"\n- {area}"
        else:
            section += """
1. Conduct experimental validation of the proposed hypothesis
2. Perform literature review for supporting evidence
3. Design experimental protocols based on proposed constraints
4. Consider computational validation where applicable
5. Identify potential collaborators in the research domain"""

        return section

    def _generate_methodology_summary(self) -> str:
        """Generate methodology summary section"""
        stats = self.execution_stats
        final_metrics = self.results.get("final_metrics", {})

        iterations = stats.get("iterations", 0)
        duration = stats.get("duration_seconds", 0)

        # Use execution_stats first (new format), fallback to final_metrics
        total_hyp = stats.get("total_hypotheses", final_metrics.get("total_hypotheses", len(self.top_hypotheses)))
        reviewed_hyp = stats.get("reviewed_hypotheses", final_metrics.get("reviewed_hypotheses", 0))
        passed_hyp = stats.get("passed_hypotheses", final_metrics.get("passed_hypotheses", 0))
        total_reviews = stats.get("total_reviews", 0)

        section = f"""## 6. Methodology Summary

### 6.1 Research Process Overview

| Metric | Value |
|--------|-------|
| Total Iterations | {iterations} |
| Duration | {duration:.1f} seconds |
| Hypotheses Generated | {total_hyp} |
| Hypotheses Reviewed | {reviewed_hyp} |
| Reviews Passed | {passed_hyp} |
| Total Reviews Conducted | {total_reviews} |

### 6.2 System Architecture

The research was conducted using BioCoScientist v3.0, which employs:

- **Multi-Agent Architecture**: Specialized agents for generation, reflection, ranking, and evolution
- **ELO Tournament System**: Pairwise hypothesis comparison using scientific debates
- **MCP Tool Integration**: 70+ domain-specific tools for data retrieval and validation
- **Adaptive Task Generation**: Dynamic workload based on research progress

### 6.3 Quality Assurance

- Multi-stage review process (initial → full → deep verification)
- Literature-grounded hypothesis generation
- Automated scientific debate for ranking
- Meta-review for cross-hypothesis insights

---

*Report generated by BioCoScientist v3.0*
*For questions or feedback, contact the development team*
"""
        return section


def generate_final_research_report(results: Dict[str, Any], output_file: str = None) -> str:
    """
    Convenience function to generate the new focused research report.

    Args:
        results: Results dict from supervisor_agent._finalize_research()
        output_file: Optional output file path

    Returns:
        Generated report text
    """
    generator = FinalResearchReportGenerator(results)
    return generator.generate(output_file)


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
