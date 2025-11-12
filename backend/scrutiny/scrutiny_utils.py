import os
import logging
from typing import List, Optional, Dict, Any
from django.db import transaction
from .models import ScrutinyResult
from .nlp_utils import analyze_file
from .ai_comparison import compare_syllabus_and_paper
from exams.models import Request

logger = logging.getLogger(__name__)

def get_existing_questions_for_subject(subject_code: str) -> List[str]:
    """
    Retrieve existing questions from previous papers for the same subject.
    Used for plagiarism detection.
    """
    try:
        # Get all final papers for the same subject
        from exams.models import FinalPapers
        previous_papers = FinalPapers.objects.filter(
            s_code=subject_code
        ).exclude(paper__isnull=True).exclude(paper='')
        
        existing_questions = []
        for paper in previous_papers:
            if paper.paper and os.path.exists(paper.paper.path):
                try:
                    analysis = analyze_file(paper.paper.path)
                    questions = analysis.get('summary', {}).get('sample_questions', [])
                    existing_questions.extend(questions)
                except Exception as e:
                    logger.warning(f"Failed to extract questions from {paper.paper.path}: {e}")
                    continue
        
        return existing_questions
    except Exception as e:
        logger.error(f"Error retrieving existing questions: {e}")
        return []

def merge_analysis_results(nlp_result: Dict[str, Any], ai_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge NLP analysis results with AI comparison results.
    Combines both analyses for comprehensive scrutiny.
    """
    merged = {
        "nlp_analysis": nlp_result.get('summary', {}),
        "ai_analysis": ai_result,
        "combined_score": 0.0,
        "recommendations": []
    }
    
    # Carry numerical fields for backward compatibility
    merged["overall_score"] = merged["combined_score"]

    # Extract scores
    nlp_score = nlp_result.get('summary', {}).get('overall_score', 0.0)
    ai_score = ai_result.get('overall_score', 0.0)
    
    # Combine scores (weighted: 40% NLP, 60% AI if both available)
    if nlp_score > 0 and ai_score > 0:
        merged["combined_score"] = round((nlp_score * 0.4) + (ai_score * 0.6), 3)
    elif ai_score > 0:
        merged["combined_score"] = ai_score
    elif nlp_score > 0:
        merged["combined_score"] = nlp_score
    
    # Combine recommendations
    nlp_recommendations = nlp_result.get('summary', {}).get('recommendations', [])
    ai_recommendations = ai_result.get('recommendations', [])
    
    # Merge and deduplicate recommendations
    all_recommendations = list(set(nlp_recommendations + ai_recommendations))
    merged["recommendations"] = all_recommendations
    
    # Add summary statistics
    merged["num_questions"] = nlp_result.get('summary', {}).get('num_questions', 0)
    merged["analysis_timestamp"] = nlp_result.get('summary', {}).get('analysis_timestamp')
    
    # Include AI-specific insights if available
    if "syllabus_coverage" in ai_result:
        merged["syllabus_coverage"] = ai_result["syllabus_coverage"]
    if "bloom_taxonomy" in ai_result:
        merged["bloom_taxonomy"] = ai_result["bloom_taxonomy"]
    if "unit_coverage" in ai_result:
        merged["unit_coverage"] = ai_result["unit_coverage"]
    if "vtu_compliance" in ai_result:
        merged["vtu_compliance"] = ai_result["vtu_compliance"]
    
    return merged


def perform_automatic_scrutiny(request_obj: Request, temp_file_path: str) -> Optional[ScrutinyResult]:
    """
    Perform automatic scrutiny analysis on an uploaded paper.
    This function is called when a teacher uploads a paper.
    Uses both NLP analysis and AI-powered syllabus comparison.
    """
    try:
        if not temp_file_path or not os.path.exists(temp_file_path):
            logger.warning(f"No temp file found for request {request_obj.id}")
            return None
        
        logger.info(f"Starting automatic scrutiny for request {request_obj.id}")
        
        # Get existing questions for plagiarism detection
        existing_questions = get_existing_questions_for_subject(request_obj.s_code)
        
        # Perform NLP-based comprehensive analysis
        nlp_analysis_result = analyze_file(temp_file_path, existing_questions)
        
        # Perform AI-powered syllabus comparison if syllabus is available
        ai_analysis_result = {}
        syllabus_path = None
        
        try:
            if request_obj.syllabus and os.path.exists(request_obj.syllabus.path):
                syllabus_path = request_obj.syllabus.path
                logger.info(f"Performing AI comparison with syllabus for request {request_obj.id}")
                ai_analysis_result = compare_syllabus_and_paper(
                    syllabus_path=syllabus_path,
                    question_paper_path=temp_file_path,
                    total_marks=getattr(request_obj, 'total_marks', 100)
                )
            else:
                logger.info(f"No syllabus available for request {request_obj.id}, skipping AI comparison")
                ai_analysis_result = {
                    "overall_score": 0.0,
                    "error": "Syllabus not available",
                    "recommendations": ["Upload syllabus file for comprehensive AI-powered analysis"]
                }
        except Exception as ai_error:
            logger.exception(f"AI comparison failed for request {request_obj.id}: {ai_error}")
            ai_analysis_result = {
                "overall_score": 0.0,
                "error": f"AI analysis failed: {str(ai_error)}",
                "recommendations": ["AI analysis unavailable. Using NLP analysis only."]
            }
        
        # Merge both analyses
        merged_result = merge_analysis_results(nlp_analysis_result, ai_analysis_result)
        
        # Create scrutiny result with merged analysis
        with transaction.atomic():
            scrutiny_result = ScrutinyResult.objects.create(
                request_obj=request_obj,
                summary=merged_result
            )
        
        logger.info(f"Automatic scrutiny completed for request {request_obj.id}. Combined score: {merged_result.get('combined_score', 0.0)}")
        return scrutiny_result
        
    except Exception as e:
        logger.exception(f"Error in automatic scrutiny for request {request_obj.id}: {e}")
        # Create a basic scrutiny result with error information
        try:
            with transaction.atomic():
                scrutiny_result = ScrutinyResult.objects.create(
                    request_obj=request_obj,
                    summary={
                        "error": str(e),
                        "num_questions": 0,
                        "overall_score": 0.0,
                        "combined_score": 0.0,
                        "recommendations": ["Analysis failed due to technical error"]
                    }
                )
            return scrutiny_result
        except Exception as create_error:
            logger.exception(f"Failed to create error scrutiny result: {create_error}")
            return None

def get_scrutiny_summary_for_dashboard() -> dict:
    """
    Get summary statistics for the COE dashboard.
    """
    try:
        total_papers = ScrutinyResult.objects.count()
        
        if total_papers == 0:
            return {
                "total_papers": 0,
                "average_score": 0.0,
                "papers_needing_review": 0,
                "plagiarism_issues": 0,
                "quality_distribution": {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
            }
        
        # Calculate statistics
        results = ScrutinyResult.objects.all()
        
        scores = []
        papers_needing_review = 0
        plagiarism_issues = 0
        quality_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for result in results:
            summary = result.summary
            # Use combined_score if available, otherwise fall back to overall_score
            score = summary.get('combined_score', summary.get('overall_score', 0.0))
            scores.append(score)
            
            # Count papers needing review (low score or plagiarism issues)
            if score < 0.6:
                papers_needing_review += 1
            
            # Count plagiarism issues
            plagiarism_score = summary.get('plagiarism_analysis', {}).get('plagiarism_score', 0.0)
            if plagiarism_score > 0.3:
                plagiarism_issues += 1
            
            # Quality distribution
            if score >= 0.8:
                quality_distribution["excellent"] += 1
            elif score >= 0.6:
                quality_distribution["good"] += 1
            elif score >= 0.4:
                quality_distribution["fair"] += 1
            else:
                quality_distribution["poor"] += 1
        
        average_score = (sum(scores) / len(scores) if scores else 0.0) * 100
        
        return {
            "total_papers": total_papers,
            "average_score": round(average_score, 2),
            "papers_needing_review": papers_needing_review,
            "plagiarism_issues": plagiarism_issues,
            "quality_distribution": quality_distribution
        }
        
    except Exception as e:
        logger.exception(f"Error getting scrutiny summary: {e}")
        return {
            "total_papers": 0,
            "average_score": 0.0,
            "papers_needing_review": 0,
            "plagiarism_issues": 0,
            "quality_distribution": {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        }
