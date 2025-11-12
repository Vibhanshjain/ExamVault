"""
Enhanced AI-powered comparison module using Google Gemini API.
Compares exam papers and ranks them based on comprehensive criteria.
"""
import os
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available. Install: pip install google-generativeai")

# Load environment variables
load_dotenv()


class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = 5
    VERY_GOOD = 4
    GOOD = 3
    FAIR = 2
    POOR = 1


@dataclass
class PaperScore:
    """Data class for exam paper scores"""
    paper_name: str
    overall_score: float
    syllabus_coverage: float
    bloom_distribution: float
    unit_coverage: float
    marks_distribution: float
    question_quality: float
    vtu_compliance: float
    recommendations: List[str]
    strengths: List[str]
    weaknesses: List[str]


# VTU Criteria weights
VTU_CRITERIA_WEIGHTS = {
    "syllabus_coverage": 0.30,
    "bloom_taxonomy_distribution": 0.25,
    "unit_coverage": 0.20,
    "marks_distribution": 0.15,
    "question_quality": 0.10
}

# Bloom's Taxonomy levels
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def initialize_gemini() -> Optional[Any]:
    """Initialize Gemini API client."""
    if not GEMINI_AVAILABLE:
        logger.error("Gemini API not available")
        return None
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini API initialized successfully")
        return model
    except Exception as e:
        logger.exception(f"Failed to initialize Gemini API: {e}")
        return None


def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or text files."""
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
                logger.warning(f"Failed to extract PDF: {e}")
                return ""
        else:
            # Read as text
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
    except Exception as e:
        logger.exception(f"Error extracting text: {e}")
        return ""


def create_comparison_prompt(syllabus_text: str, question_paper_text: str, total_marks: int = 100) -> str:
    """Create comprehensive analysis prompt for Gemini."""
    prompt = f"""You are an expert academic evaluator specializing in VTU exam standards.
Analyze this exam paper against its syllabus comprehensively.

SYLLABUS:
{syllabus_text[:5000]}

QUESTION PAPER:
{question_paper_text[:5000]}

TOTAL MARKS: {total_marks}

Provide JSON analysis with:
1. Syllabus coverage (score 0-1, covered/missing topics)
2. Bloom's taxonomy distribution (Remember to Create)
3. Unit-wise coverage (all units)
4. Marks distribution analysis
5. Question quality assessment
6. VTU compliance check
7. Overall score and recommendations

Return ONLY valid JSON."""
    
    return prompt


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse Gemini's JSON response."""
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        # Find JSON boundaries
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            response_text = response_text[start_idx:end_idx]
        
        result = json.loads(response_text)
        return result
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        return {"error": str(e), "overall_score": 0.0}


def analyze_single_paper(syllabus_path: str, paper_path: str, paper_name: str, total_marks: int = 100) -> PaperScore:
    """Analyze a single exam paper."""
    model = initialize_gemini()
    if not model:
        raise RuntimeError("Gemini API not available")
    
    # Extract text
    syllabus_text = extract_text_from_file(syllabus_path) if syllabus_path else ""
    paper_text = extract_text_from_file(paper_path)
    
    if not paper_text:
        raise ValueError(f"Could not read paper: {paper_path}")
    
    # Create prompt and get response
    prompt = create_comparison_prompt(syllabus_text, paper_text, total_marks)
    response = model.generate_content(prompt)
    analysis = parse_gemini_response(response.text)
    
    # Extract scores
    return PaperScore(
        paper_name=paper_name,
        overall_score=analysis.get("overall_score", 0.0),
        syllabus_coverage=analysis.get("syllabus_coverage", {}).get("score", 0.0),
        bloom_distribution=analysis.get("bloom_taxonomy", {}).get("score", 0.0),
        unit_coverage=analysis.get("unit_coverage", {}).get("score", 0.0),
        marks_distribution=analysis.get("marks_distribution", {}).get("score", 0.0),
        question_quality=analysis.get("question_quality", {}).get("score", 0.0),
        vtu_compliance=analysis.get("vtu_compliance", {}).get("score", 0.0),
        recommendations=analysis.get("recommendations", []),
        strengths=analysis.get("vtu_compliance", {}).get("strengths", []),
        weaknesses=analysis.get("vtu_compliance", {}).get("issues", [])
    )


def compare_multiple_papers(syllabus_path: str, paper_paths: List[Tuple[str, str]], total_marks: int = 100) -> List[PaperScore]:
    """Compare multiple exam papers and rank them."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini API not available")
    
    results = []
    for paper_path, paper_name in paper_paths:
        try:
            score = analyze_single_paper(syllabus_path, paper_path, paper_name, total_marks)
            results.append(score)
            logger.info(f"Analyzed {paper_name}: {score.overall_score}")
        except Exception as e:
            logger.error(f"Failed to analyze {paper_name}: {e}")
            continue
    
    # Sort by overall score (descending)
    results.sort(key=lambda x: x.overall_score, reverse=True)
    return results


def get_best_paper(papers: List[PaperScore]) -> Optional[PaperScore]:
    """Get the best paper from the list."""
    return papers[0] if papers else None


def export_comparison_results(papers: List[PaperScore], output_file: str) -> None:
    """Export comparison results to JSON."""
    results = {
        "papers": [
            {
                "name": p.paper_name,
                "overall_score": p.overall_score,
                "syllabus_coverage": p.syllabus_coverage,
                "bloom_distribution": p.bloom_distribution,
                "unit_coverage": p.unit_coverage,
                "marks_distribution": p.marks_distribution,
                "question_quality": p.question_quality,
                "vtu_compliance": p.vtu_compliance,
                "strengths": p.strengths,
                "weaknesses": p.weaknesses,
                "recommendations": p.recommendations
            }
            for p in papers
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results exported to {output_file}")
