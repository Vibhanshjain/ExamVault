import os
import re
import logging
import hashlib
import difflib
from collections import Counter
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Check if required libraries are available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some features will be limited.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Some features will be limited.")

logger = logging.getLogger(__name__)

# Bloom's Taxonomy keywords for classification
BLOOM_KEYWORDS = {
    'remember': ['define', 'identify', 'list', 'name', 'recall', 'recognize', 'state', 'what', 'when', 'where', 'who'],
    'understand': ['explain', 'describe', 'interpret', 'summarize', 'classify', 'compare', 'contrast', 'demonstrate'],
    'apply': ['apply', 'use', 'solve', 'demonstrate', 'illustrate', 'calculate', 'compute', 'implement'],
    'analyze': ['analyze', 'examine', 'investigate', 'compare', 'contrast', 'differentiate', 'distinguish'],
    'evaluate': ['evaluate', 'judge', 'critique', 'assess', 'appraise', 'justify', 'defend', 'argue'],
    'create': ['create', 'design', 'develop', 'construct', 'formulate', 'generate', 'produce', 'build']
}

# Difficulty indicators
DIFFICULTY_INDICATORS = {
    'easy': ['simple', 'basic', 'easy', 'straightforward', 'obvious', 'clear'],
    'medium': ['moderate', 'intermediate', 'standard', 'typical', 'common'],
    'hard': ['complex', 'difficult', 'challenging', 'advanced', 'sophisticated', 'intricate']
}

# Target distributions inspired by VTU blueprints (balanced across modules and higher order skills)
TARGET_BLOOM_DISTRIBUTION = {
    'remember': 0.15,
    'understand': 0.25,
    'apply': 0.20,
    'analyze': 0.15,
    'evaluate': 0.15,
    'create': 0.10,
}
HIGHER_ORDER_LEVELS = ['analyze', 'evaluate', 'create']
MIN_HIGHER_ORDER_RATIO = 0.35
MAX_REMEMBER_RATIO = 0.25

TARGET_DIFFICULTY_DISTRIBUTION = {
    'easy': 0.2,
    'medium': 0.5,
    'hard': 0.3,
}

EXPECTED_QUESTION_COUNT = 10  # VTU theory papers typically expect ~10 full questions/options

def extract_text_from_pdf(path):
    """
    Try PyMuPDF if available, otherwise fallback to simple read (may not work for binary pdfs).
    """
    try:
        import fitz
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text("text"))
        return "\n".join(text)
    except Exception as e:
        logger.warning("extract_text_from_pdf: fitz not available or failed: %s", e)
        try:
            with open(path, "rb") as f:
                raw = f.read()
                try:
                    return raw.decode("latin1")
                except Exception:
                    return ""
        except Exception:
            return ""

QUESTION_SPLIT_PATTERNS = [
    r"(?:^|\n)\s*(?:Q[\.\-]?\s*\d{1,2}[A-Z]?)",
    r"(?:^|\n)\s*(?:Module\s*[-–]?\s*\d+)",
]

def split_into_questions(text):
    """
    Improved heuristic: split on explicit VTU-style numbering (Q.01, Module-1) and question marks.
    """
    if not text:
        return []
    # Normalize newlines
    text = re.sub(r'\r\n', '\n', text)

    # First split on VTU question numbering patterns.
    # We insert a delimiter before question identifiers to aid splitting.
    processed = text
    for pattern in QUESTION_SPLIT_PATTERNS:
        processed = re.sub(pattern, lambda m: "\n###SPLIT###" + m.group(0).strip(), processed, flags=re.IGNORECASE)

    candidates = [c.strip() for c in processed.split("###SPLIT###") if c.strip()]

    questions = []
    seen = set()
    for candidate in candidates:
        # Try to pull up to the first double newline for cleaner chunk
        candidate = candidate.strip()
        if not candidate:
            continue
        # If it does not end with question mark, attempt to trim at question mark.
        if '?' in candidate:
            parts = candidate.split('?')
            for idx, part in enumerate(parts[:-1]):
                q = (part + '?').strip()
                if len(q.split()) >= 4 and q not in seen:
                    seen.add(q)
                    questions.append(q)
            tail = parts[-1].strip()
            if tail and len(tail.split()) >= 6 and tail not in seen:
                seen.add(tail)
                questions.append(tail)
        else:
            if len(candidate.split()) >= 6 and candidate not in seen:
                seen.add(candidate)
                questions.append(candidate)

    # As a fallback, split on plain question marks if we detected too few questions.
    # Keep only pieces that look like questions or are substantial lines
    if len(questions) < 5:
        fallback_parts = [p.strip() + "?" for p in text.split("?") if p.strip()]
        for p in fallback_parts:
            p = re.sub(r'\s+', ' ', p).strip()
            if len(p.split()) >= 4 and p.endswith("?") and p not in seen:
                seen.add(p)
                questions.append(p)

    return questions[:50]

def classify_bloom_taxonomy(question: str) -> Dict[str, float]:
    """
    Classify question based on Bloom's Taxonomy using keyword matching and NLP.
    Returns confidence scores for each level.
    """
    question_lower = question.lower()
    scores = {level: 0.0 for level in BLOOM_KEYWORDS.keys()}
    
    # Keyword-based scoring
    for level, keywords in BLOOM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question_lower:
                scores[level] += 1.0
    
    # Normalize scores
    total_score = sum(scores.values())
    if total_score > 0:
        scores = {k: v/total_score for k, v in scores.items()}
    
    return scores

def estimate_difficulty(question: str) -> Dict[str, Any]:
    """
    Estimate question difficulty based on various linguistic features.
    """
    difficulty_score = 0.5  # Default medium difficulty
    
    question_lower = question.lower()
    
    # Check for difficulty indicators
    for level, indicators in DIFFICULTY_INDICATORS.items():
        for indicator in indicators:
            if indicator in question_lower:
                if level == 'easy':
                    difficulty_score -= 0.2
                elif level == 'hard':
                    difficulty_score += 0.2
    
    # Sentence length analysis
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(question)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        except:
            # Fallback to simple sentence counting
            sentences = question.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    else:
        # Fallback to simple sentence counting
        sentences = question.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    if avg_sentence_length > 20:
        difficulty_score += 0.1
    elif avg_sentence_length < 10:
        difficulty_score -= 0.1
    
    # Word complexity (average word length)
    if NLTK_AVAILABLE:
        try:
            words = word_tokenize(question)
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        except:
            # Fallback to simple word splitting
            words = question.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    else:
        # Fallback to simple word splitting
        words = question.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    if avg_word_length > 6:
        difficulty_score += 0.1
    elif avg_word_length < 4:
        difficulty_score -= 0.1
    
    # Clamp between 0 and 1
    difficulty_score = max(0, min(1, difficulty_score))
    
    # Convert to categorical
    if difficulty_score < 0.33:
        difficulty_level = "easy"
    elif difficulty_score < 0.66:
        difficulty_level = "medium"
    else:
        difficulty_level = "hard"
    
    return {
        "level": difficulty_level,
        "score": difficulty_score,
        "features": {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length
        }
    }

def detect_plagiarism_and_duplicates(questions: List[str], existing_questions: List[str] = None) -> Dict[str, Any]:
    """
    Detect potential plagiarism and duplicate questions using text similarity.
    """
    plagiarism_results = {
        "duplicates": [],
        "similar_questions": [],
        "plagiarism_score": 0.0
    }
    
    if len(questions) < 2:
        return plagiarism_results
    
    # Check for duplicates within the same paper
    for i, q1 in enumerate(questions):
        for j, q2 in enumerate(questions[i+1:], i+1):
            similarity = difflib.SequenceMatcher(None, q1.lower(), q2.lower()).ratio()
            if similarity > 0.8:
                plagiarism_results["duplicates"].append({
                    "question1_index": i,
                    "question2_index": j,
                    "similarity": similarity,
                    "question1": q1[:100] + "..." if len(q1) > 100 else q1,
                    "question2": q2[:100] + "..." if len(q2) > 100 else q2
                })
    
    # Check against existing questions if provided
    if existing_questions and SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            for i, question in enumerate(questions):
                similarities = []
                for existing_q in existing_questions:
                    try:
                        tfidf_matrix = vectorizer.fit_transform([question, existing_q])
                        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        similarities.append(similarity)
                    except:
                        continue
                
                if similarities:
                    max_similarity = max(similarities)
                    if max_similarity > 0.7:
                        plagiarism_results["similar_questions"].append({
                            "question_index": i,
                            "similarity": max_similarity,
                            "question": question[:100] + "..." if len(question) > 100 else question
                        })
        except Exception as e:
            logger.warning(f"Advanced plagiarism detection failed: {e}")
    
    # Calculate overall plagiarism score
    total_questions = len(questions)
    duplicate_count = len(plagiarism_results["duplicates"])
    similar_count = len(plagiarism_results["similar_questions"])
    
    plagiarism_results["plagiarism_score"] = (duplicate_count + similar_count) / max(total_questions, 1)
    
    return plagiarism_results

def extract_question_tags(question: str) -> List[str]:
    """
    Extract meaningful tags from a question using NLP.
    """
    if NLTK_AVAILABLE:
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(question.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract nouns and adjectives as potential tags
            tags = []
            for word, pos in pos_tags:
                if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'] and len(word) > 3:
                    tags.append(word)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tags = [tag for tag in tags if tag not in stop_words]
            
            # Return top 5 most relevant tags
            return list(set(tags))[:5]
        except Exception as e:
            logger.warning(f"NLTK tag extraction failed: {e}")
    
    # Fallback to simple word extraction
    words = re.findall(r'\b[A-Za-z]{4,}\b', question.lower())
    stop_words = set(["the", "and", "for", "with", "that", "this", "from", "have", "are", "using", "use", "what", "how", "why", "when", "where"])
    return [w for w in words if w not in stop_words][:5]

def evaluate_bloom_distribution(bloom_distribution: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Compare observed Bloom distribution with target distribution and return
    an alignment score in [0,1] along with actionable recommendations.
    """
    # Calculate L1 distance against target profile
    diff = 0.0
    for level, target in TARGET_BLOOM_DISTRIBUTION.items():
        diff += abs(bloom_distribution.get(level, 0.0) - target)
    alignment_score = max(0.0, 1.0 - diff / 2.0)  # diff is at most 2.0

    higher_order_ratio = sum(bloom_distribution.get(level, 0.0) for level in HIGHER_ORDER_LEVELS)
    recommendations: List[str] = []

    if higher_order_ratio < MIN_HIGHER_ORDER_RATIO:
        alignment_score -= (MIN_HIGHER_ORDER_RATIO - higher_order_ratio) * 0.6
        recommendations.append(
            "Add more higher-order questions (analyze/evaluate/create) to align with Bloom's expectations."
        )

    remember_ratio = bloom_distribution.get('remember', 0.0)
    if remember_ratio > MAX_REMEMBER_RATIO + 0.1:
        alignment_score -= (remember_ratio - MAX_REMEMBER_RATIO) * 0.5
        recommendations.append("Reduce straightforward recall questions; diversify into application and analysis levels.")

    understand_ratio = bloom_distribution.get('understand', 0.0)
    if understand_ratio > 0.4:
        alignment_score -= (understand_ratio - 0.4) * 0.3
        recommendations.append("Limit descriptive 'understand' questions and shift towards problem-solving or design tasks.")

    alignment_score = max(0.0, min(1.0, alignment_score))
    return alignment_score, recommendations


def evaluate_difficulty_distribution(difficulty_distribution: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Evaluate question difficulty spread against a balanced blueprint.
    Returns alignment score in [0,1] and recommendations.
    """
    diff = 0.0
    for level, target in TARGET_DIFFICULTY_DISTRIBUTION.items():
        diff += abs(difficulty_distribution.get(level, 0.0) - target)
    alignment_score = max(0.0, 1.0 - diff / 2.0)

    recommendations: List[str] = []
    easy_ratio = difficulty_distribution.get('easy', 0.0)
    hard_ratio = difficulty_distribution.get('hard', 0.0)

    if easy_ratio > 0.35:
        alignment_score -= (easy_ratio - 0.35) * 0.5
        recommendations.append("Too many easy questions. Introduce tougher analytical or design-oriented problems.")
    if hard_ratio < 0.2:
        alignment_score -= (0.2 - hard_ratio) * 0.4
        recommendations.append("Increase higher difficulty questions to challenge top-performing students.")

    alignment_score = max(0.0, min(1.0, alignment_score))
    return alignment_score, recommendations


def compute_integrity_score(plagiarism_analysis: Dict[str, Any], num_questions: int) -> Tuple[float, List[str]]:
    """
    Combine plagiarism, duplicate, and similarity information into a single integrity score.
    """
    plagiarism_score = plagiarism_analysis.get("plagiarism_score", 0.0)
    duplicate_penalty = len(plagiarism_analysis.get("duplicates", [])) / max(1, num_questions)
    similar_penalty = len(plagiarism_analysis.get("similar_questions", [])) / max(1, num_questions)

    integrity_score = 1.0 - min(1.0, plagiarism_score * 0.7 + duplicate_penalty * 0.5 + similar_penalty * 0.3)
    integrity_score = max(0.0, min(1.0, integrity_score))

    recommendations: List[str] = []
    if plagiarism_score > 0.25:
        recommendations.append("High similarity to previous papers detected. Revise or rephrase overlapping questions.")
    if duplicate_penalty > 0:
        recommendations.append("Found duplicate/similar questions within the paper. Replace with unique scenarios.")

    return integrity_score, recommendations

def evaluate_structural_criteria(raw_text: str, num_questions: int) -> Tuple[float, List[str]]:
    """
    Evaluate VTU structural compliance heuristically.
    """
    score = 1.0
    recommendations: List[str] = []
    lowered = raw_text.lower()

    if num_questions < EXPECTED_QUESTION_COUNT:
        score -= 0.25
        recommendations.append(f"Increase question count to at least {EXPECTED_QUESTION_COUNT} full questions.")

    if "max. marks" not in lowered and "max marks" not in lowered:
        score -= 0.1
        recommendations.append("Mention total marks (e.g., 'Max. Marks: 100').")

    if "time" not in lowered or "hours" not in lowered:
        score -= 0.05
        recommendations.append("Include examination duration (e.g., 'Time: 03 Hours').")

    module_mentions = len(re.findall(r'module\s*[-–]?\s*\d', lowered))
    if module_mentions < 5:
        score -= 0.2
        recommendations.append("Ensure each module is clearly delineated with headings (Module 1 ... Module 5).")

    if "answer any" not in lowered:
        score -= 0.05
        recommendations.append("Clarify instructions (e.g., 'Answer any FIVE full questions, choosing one from each module').")

    practical_keywords = ['virtual machine', 'cloud shell', 'experiment', 'practical']
    if not any(keyword in lowered for keyword in practical_keywords):
        score -= 0.05
        recommendations.append("Include at least one practical-oriented question from the lab component.")

    score = max(0.0, min(1.0, score))
    return score, recommendations

def analyze_file(path, existing_questions: List[str] = None):
    """
    Comprehensive analysis with NLP/ML features for automatic scrutiny.
    Returns detailed analysis including Bloom taxonomy, difficulty, plagiarism detection.
    """
    logger.debug("scrutiny.analyze_file called for path=%s", path)
    
    analysis_result = {
        "summary": {
            "num_questions": 0,
            "sample_questions": [],
            "tags": [],
            "notes": [],
            "bloom_distribution": {},
            "difficulty_distribution": {},
            "plagiarism_analysis": {},
            "overall_score": 0.0,
            "recommendations": []
        }
    }
    
    try:
        ext = os.path.splitext(path)[1].lower()
        text = ""
        if ext in (".pdf",):
            text = extract_text_from_pdf(path)
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = extract_text_from_pdf(path)

        questions = split_into_questions(text)
        analysis_result["summary"]["num_questions"] = len(questions)
        analysis_result["summary"]["sample_questions"] = questions[:5]
        
        if not questions:
            analysis_result["summary"]["notes"].append("No questions detected in the document")
            return analysis_result
        
        # Analyze each question
        bloom_scores = []
        difficulty_scores = []
        all_tags = []
        
        for question in questions:
            # Bloom taxonomy classification
            bloom_result = classify_bloom_taxonomy(question)
            bloom_scores.append(bloom_result)
            
            # Difficulty estimation
            difficulty_result = estimate_difficulty(question)
            difficulty_scores.append(difficulty_result)
            
            # Tag extraction
            tags = extract_question_tags(question)
            all_tags.extend(tags)
        
        # Calculate distributions
        bloom_distribution = {}
        for level in BLOOM_KEYWORDS.keys():
            bloom_distribution[level] = sum(score[level] for score in bloom_scores) / len(bloom_scores)
        
        difficulty_distribution = {}
        for level in ['easy', 'medium', 'hard']:
            difficulty_distribution[level] = sum(1 for d in difficulty_scores if d['level'] == level) / len(difficulty_scores)
        
        # Plagiarism detection
        plagiarism_analysis = detect_plagiarism_and_duplicates(questions, existing_questions)

        # Evaluate alignment metrics
        bloom_alignment_score, bloom_recs = evaluate_bloom_distribution(bloom_distribution)
        difficulty_alignment_score, difficulty_recs = evaluate_difficulty_distribution(difficulty_distribution)
        integrity_score, integrity_recs = compute_integrity_score(plagiarism_analysis, len(questions))
        coverage_score = min(1.0, len(questions) / EXPECTED_QUESTION_COUNT)
        structural_score, structural_recs = evaluate_structural_criteria(text, len(questions))

        # Calculate overall quality using weighted components
        bloom_weight = 0.4
        difficulty_weight = 0.25
        coverage_weight = 0.2
        integrity_weight = 0.1
        structure_weight = 0.05

        weighted_quality_score = (
            bloom_weight * bloom_alignment_score +
            difficulty_weight * difficulty_alignment_score +
            coverage_weight * coverage_score +
            integrity_weight * integrity_score +
            structure_weight * structural_score
        )
        weighted_quality_score = max(0.0, min(1.0, weighted_quality_score))

        # Generate recommendations
        recommendations: List[str] = []
        recommendations.extend(bloom_recs)
        recommendations.extend(difficulty_recs)
        recommendations.extend(integrity_recs)
        recommendations.extend(structural_recs)
        if coverage_score < 0.6:
            recommendations.append("Increase the number of distinct questions to cover the full syllabus blueprint.")
        if len(questions) < EXPECTED_QUESTION_COUNT:
            analysis_result["summary"]["notes"].append(
                f"Detected only {len(questions)} questions. VTU papers typically include ~{EXPECTED_QUESTION_COUNT}."
            )
        if quality_score < 0.6:
            recommendations.append("Overall paper balance is weak. Revisit blueprint coverage and cognitive levels.")
        
        syllabus_coverage_score_10 = round(coverage_score * 10, 2)
        bloom_taxonomy_score_10 = round(bloom_alignment_score * 10, 2)
        structural_score_10 = round(structural_score * 10, 2)
        vt_score_components = [syllabus_coverage_score_10, bloom_taxonomy_score_10, structural_score_10]
        overall_score_10 = round(sum(vt_score_components) / len(vt_score_components), 2)
        normalized_overall = round(overall_score_10 / 10.0, 3)

        # Compile final results
        analysis_result["summary"].update({
            "bloom_distribution": bloom_distribution,
            "difficulty_distribution": difficulty_distribution,
            "plagiarism_analysis": plagiarism_analysis,
            "bloom_alignment_score": round(bloom_alignment_score, 3),
            "difficulty_alignment_score": round(difficulty_alignment_score, 3),
            "coverage_score": round(coverage_score, 3),
            "integrity_score": round(integrity_score, 3),
            "structural_score": round(structural_score, 3),
            "score_breakdown": {
                "bloom": round(bloom_weight * bloom_alignment_score, 3),
                "difficulty": round(difficulty_weight * difficulty_alignment_score, 3),
                "coverage": round(coverage_weight * coverage_score, 3),
                "integrity": round(integrity_weight * integrity_score, 3),
                "structure": round(structure_weight * structural_score, 3)
            },
            "syllabus_coverage_score_10": syllabus_coverage_score_10,
            "bloom_taxonomy_score_10": bloom_taxonomy_score_10,
            "structural_score_10": structural_score_10,
            "overall_score_10": overall_score_10,
            "overall_score": normalized_overall,
            "recommendations": list(dict.fromkeys(recommendations)),
            "tags": list(set(all_tags))[:10]  # Top 10 unique tags
        })
        
        return analysis_result
        
    except Exception as e:
        logger.exception("scrutiny.analyze_file failed: %s", e)
        analysis_result["summary"]["error"] = str(e)
        return analysis_result
