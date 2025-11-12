"""
AI-powered comparison module using Google Gemini API to compare syllabus and question papers.
Evaluates based on Bloom's Taxonomy, VTU criteria, and syllabus coverage.
"""
import os
import logging
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available. AI comparison features will be disabled.")

# Load environment variables
load_dotenv()

# Bloom's Taxonomy levels for VTU evaluation
BLOOM_LEVELS = [
    "Remember",
    "Understand", 
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

# VTU Criteria weights (can be adjusted)
VTU_CRITERIA_WEIGHTS = {
    "syllabus_coverage": 0.30,  # 30% weight
    "bloom_taxonomy_distribution": 0.25,  # 25% weight
    "unit_coverage": 0.20,  # 20% weight
    "marks_distribution": 0.15,  # 15% weight
    "question_quality": 0.10  # 10% weight
}


def initialize_gemini() -> Optional[Any]:
    """
    Initialize Gemini API client.
    Returns the model instance if successful, None otherwise.
    """
    if not GEMINI_AVAILABLE:
        logger.error("Gemini API not available. Install google-generativeai package.")
        return None
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini API initialized successfully")
        return model
    except Exception as e:
        logger.exception(f"Failed to initialize Gemini API: {e}")
        return None


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from PDF or text file.
    """
    if not file_path or not os.path.exists(file_path):
        return ""
    
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = []
                for page in doc:
                    text.append(page.get_text("text"))
                return "\n".join(text)
            except Exception as e:
                logger.warning(f"Failed to extract PDF text: {e}")
                return ""
        else:
            # Try to read as text file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
    except Exception as e:
        logger.exception(f"Error extracting text from {file_path}: {e}")
        return ""


def create_comparison_prompt(syllabus_text: str, question_paper_text: str, total_marks: int = 100) -> str:
    """
    Create a comprehensive prompt for Gemini to compare syllabus and question paper.
    """
    prompt = f"""You are an expert academic evaluator for Visvesvaraya Technological University (VTU). 
Your task is to analyze a question paper against its syllabus and provide a comprehensive evaluation.

SYLLABUS CONTENT:
{syllabus_text[:8000]}  # Limit to avoid token limits

QUESTION PAPER CONTENT:
{question_paper_text[:8000]}  # Limit to avoid token limits

TOTAL MARKS: {total_marks}

Please provide a detailed analysis in the following JSON format:

{{
    "syllabus_coverage": {{
        "score": 0.0-1.0,
        "covered_topics": ["list of topics covered"],
        "missing_topics": ["list of important topics not covered"],
        "coverage_percentage": 0.0-100.0,
        "analysis": "detailed explanation"
    }},
    "bloom_taxonomy": {{
        "distribution": {{
            "remember": 0.0-1.0,
            "understand": 0.0-1.0,
            "apply": 0.0-1.0,
            "analyze": 0.0-1.0,
            "evaluate": 0.0-1.0,
            "create": 0.0-1.0
        }},
        "score": 0.0-1.0,
        "recommendations": ["suggestions for better distribution"],
        "analysis": "detailed explanation of Bloom's taxonomy alignment"
    }},
    "unit_coverage": {{
        "score": 0.0-1.0,
        "units_covered": {{
            "unit_1": {{"covered": true/false, "marks": 0, "topics": ["list"]}},
            "unit_2": {{"covered": true/false, "marks": 0, "topics": ["list"]}},
            "unit_3": {{"covered": true/false, "marks": 0, "topics": ["list"]}},
            "unit_4": {{"covered": true/false, "marks": 0, "topics": ["list"]}},
            "unit_5": {{"covered": true/false, "marks": 0, "topics": ["list"]}}
        }},
        "analysis": "detailed explanation of unit-wise coverage"
    }},
    "marks_distribution": {{
        "score": 0.0-1.0,
        "distribution": {{
            "section_a": {{"marks": 0, "questions": 0, "percentage": 0.0}},
            "section_b": {{"marks": 0, "questions": 0, "percentage": 0.0}},
            "section_c": {{"marks": 0, "questions": 0, "percentage": 0.0}}
        }},
        "analysis": "evaluation of marks distribution across sections"
    }},
    "question_quality": {{
        "score": 0.0-1.0,
        "clarity": 0.0-1.0,
        "relevance": 0.0-1.0,
        "difficulty_appropriateness": 0.0-1.0,
        "analysis": "detailed quality assessment"
    }},
    "vtu_compliance": {{
        "score": 0.0-1.0,
        "issues": ["list of VTU compliance issues"],
        "strengths": ["list of strengths"],
        "analysis": "overall VTU compliance evaluation"
    }},
    "overall_score": 0.0-1.0,
    "recommendations": ["list of actionable recommendations"],
    "detailed_feedback": "comprehensive feedback for improvement"
}}

IMPORTANT INSTRUCTIONS:
1. Analyze the syllabus to identify all units and topics
2. Map each question in the paper to syllabus topics and Bloom's taxonomy levels
3. Check if all units are adequately covered (VTU typically requires coverage of all units)
4. Evaluate marks distribution (VTU typically follows specific patterns like 20-30-50 or similar)
5. Assess question quality in terms of clarity, relevance, and appropriateness
6. Provide specific, actionable recommendations
7. Return ONLY valid JSON, no additional text before or after

Ensure the analysis is thorough and follows VTU examination standards."""
    
    return prompt


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """
    Parse Gemini's response and extract JSON.
    Handles cases where response might have markdown code blocks or extra text.
    """
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        # Try to find JSON object boundaries
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            response_text = response_text[start_idx:end_idx]
        
        # Parse JSON
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        # Return a default structure with error
        return {
            "error": f"Failed to parse AI response: {str(e)}",
            "overall_score": 0.0,
            "recommendations": ["AI analysis failed. Please review manually."]
        }
    except Exception as e:
        logger.exception(f"Unexpected error parsing response: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "overall_score": 0.0,
            "recommendations": ["AI analysis failed. Please review manually."]
        }


def compare_syllabus_and_paper(
    syllabus_path: Optional[str],
    question_paper_path: str,
    total_marks: int = 100
) -> Dict[str, Any]:
    """
    Main function to compare syllabus and question paper using Gemini AI.
    
    Args:
        syllabus_path: Path to syllabus file (PDF or text)
        question_paper_path: Path to question paper file (PDF or text)
        total_marks: Total marks for the paper
    
    Returns:
        Dictionary containing comprehensive analysis results
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini API not available. Returning empty analysis.")
        return {
            "error": "Gemini API not available. Install google-generativeai package.",
            "overall_score": 0.0,
            "recommendations": ["AI analysis unavailable. Please review manually."]
        }
    
    # Initialize Gemini
    model = initialize_gemini()
    if not model:
        return {
            "error": "Failed to initialize Gemini API. Check GEMINI_API_KEY.",
            "overall_score": 0.0,
            "recommendations": ["AI analysis unavailable. Please review manually."]
        }
    
    # Extract text from files
    syllabus_text = ""
    if syllabus_path and os.path.exists(syllabus_path):
        syllabus_text = extract_text_from_file(syllabus_path)
        if not syllabus_text:
            logger.warning(f"Could not extract text from syllabus: {syllabus_path}")
    else:
        logger.warning("Syllabus path not provided or file does not exist")
    
    question_paper_text = extract_text_from_file(question_paper_path)
    if not question_paper_text:
        logger.error(f"Could not extract text from question paper: {question_paper_path}")
        return {
            "error": "Could not extract text from question paper",
            "overall_score": 0.0,
            "recommendations": ["Failed to read question paper. Please check file format."]
        }
    
    if not syllabus_text:
        logger.warning("No syllabus text available. Analysis will be limited.")
        # Still analyze the paper, but note syllabus is missing
        return {
            "error": "Syllabus not available for comparison",
            "overall_score": 0.0,
            "recommendations": ["Syllabus file missing. Upload syllabus for comprehensive analysis."],
            "question_paper_analysis": {
                "num_questions": len(question_paper_text.split("?")) if "?" in question_paper_text else 0,
                "text_length": len(question_paper_text)
            }
        }
    
    try:
        # Create prompt
        prompt = create_comparison_prompt(syllabus_text, question_paper_text, total_marks)
        
        # Call Gemini API
        logger.info("Calling Gemini API for syllabus-paper comparison...")
        response = model.generate_content(prompt)
        
        # Extract response text
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse response
        analysis_result = parse_gemini_response(response_text)
        
        # Calculate weighted overall score if not provided
        if "overall_score" not in analysis_result or analysis_result["overall_score"] == 0.0:
            overall_score = calculate_weighted_score(analysis_result)
            analysis_result["overall_score"] = overall_score
        
        # Add metadata
        analysis_result["ai_model"] = "gemini-pro"
        analysis_result["analysis_type"] = "syllabus_paper_comparison"
        
        logger.info(f"AI comparison completed. Overall score: {analysis_result.get('overall_score', 0.0)}")
        return analysis_result
        
    except Exception as e:
        logger.exception(f"Error during AI comparison: {e}")
        return {
            "error": f"AI analysis failed: {str(e)}",
            "overall_score": 0.0,
            "recommendations": ["AI analysis encountered an error. Please review manually."]
        }


def calculate_weighted_score(analysis_result: Dict[str, Any]) -> float:
    """
    Calculate weighted overall score based on VTU criteria.
    """
    try:
        scores = {}
        weights = VTU_CRITERIA_WEIGHTS
        
        # Extract scores from analysis result
        if "syllabus_coverage" in analysis_result:
            scores["syllabus_coverage"] = analysis_result["syllabus_coverage"].get("score", 0.0)
        
        if "bloom_taxonomy" in analysis_result:
            scores["bloom_taxonomy_distribution"] = analysis_result["bloom_taxonomy"].get("score", 0.0)
        
        if "unit_coverage" in analysis_result:
            scores["unit_coverage"] = analysis_result["unit_coverage"].get("score", 0.0)
        
        if "marks_distribution" in analysis_result:
            scores["marks_distribution"] = analysis_result["marks_distribution"].get("score", 0.0)
        
        if "question_quality" in analysis_result:
            scores["question_quality"] = analysis_result["question_quality"].get("score", 0.0)
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, weight in weights.items():
            if criterion in scores:
                weighted_sum += scores[criterion] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.0
        
        return round(overall_score, 3)
    except Exception as e:
        logger.exception(f"Error calculating weighted score: {e}")
        return 0.0


