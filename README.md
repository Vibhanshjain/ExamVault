# ExamVault - AI-Powered Exam Paper Comparison System

## Overview

ExamVault is an intelligent system for comparing and ranking exam papers using advanced AI powered by Google Gemini API. It evaluates exam papers based on VTU (Visvesvaraya Technological University) standards, Bloom's taxonomy, syllabus coverage, and multiple quality metrics.

## Features

### Core Functionality
- **AI-Powered Comparison**: Uses Google Gemini API for intelligent analysis
- **Multiple Paper Ranking**: Compare multiple exam papers and automatically rank them
- **Comprehensive Metrics**:
  - Syllabus Coverage Analysis
  - Bloom's Taxonomy Distribution
  - Unit-wise Coverage
  - Marks Distribution
  - Question Quality Assessment
  - VTU Compliance Check

### Key Components

#### 1. **ai_comparison.py** - Core AI Module
Main module containing all comparison logic:
- `analyze_single_paper()`: Analyze a single exam paper
- `compare_multiple_papers()`: Compare and rank multiple papers
- `export_comparison_results()`: Export results to JSON
- `PaperScore`: Data class for storing paper scores

#### 2. **comparison_api.py** - REST API Endpoints
Django REST Framework ViewSet providing:
- `POST /compare/`: Compare multiple papers
- `POST /analyze/`: Analyze a single paper

#### 3. **requirements.txt** - Dependencies
Includes:
- Django & DRF
- Google Generative AI
- PyMuPDF for PDF processing
- NLP & ML libraries
- Data validation libraries

## Installation

### Prerequisites
- Python 3.8+
- Django 4.2+
- Google Gemini API Key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Vibhanshjain/ExamVault.git
cd ExamVault/backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables**
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

4. **Run migrations**
```bash
python manage.py migrate
```

5. **Start the server**
```bash
python manage.py runserver
```

## Usage

### Python API Usage

```python
from scrutiny.ai_comparison import compare_multiple_papers

# Define papers to compare
papers = [
    ('path/to/paper1.pdf', 'Paper 2024-A'),
    ('path/to/paper2.pdf', 'Paper 2024-B')
]

# Compare papers
results = compare_multiple_papers(
    syllabus_path='path/to/syllabus.pdf',
    paper_paths=papers,
    total_marks=100
)

# Get best paper
best_paper = results[0]  # Already sorted by score
print(f"Best Paper: {best_paper.paper_name}")
print(f"Score: {best_paper.overall_score}")
```

### REST API Usage

#### Compare Multiple Papers
```bash
curl -X POST http://localhost:8000/api/comparison/compare/ \
  -H "Content-Type: application/json" \
  -d '{
    "syllabus_path": "path/to/syllabus.pdf",
    "papers": [
      {"path": "path/to/paper1.pdf", "name": "Paper 2024-A"},
      {"path": "path/to/paper2.pdf", "name": "Paper 2024-B"}
    ],
    "total_marks": 100
  }'
```

#### Analyze Single Paper
```bash
curl -X POST http://localhost:8000/api/comparison/analyze/ \
  -H "Content-Type: application/json" \
  -d '{
    "syllabus_path": "path/to/syllabus.pdf",
    "paper_path": "path/to/paper.pdf",
    "paper_name": "Paper 2024",
    "total_marks": 100
  }'
```

## API Response Format

### Comparison Response
```json
{
  "status": "success",
  "papers_analyzed": 2,
  "best_paper": {
    "name": "Paper 2024-A",
    "overall_score": 0.85,
    "syllabus_coverage": 0.90,
    "bloom_distribution": 0.82,
    "unit_coverage": 0.88,
    "marks_distribution": 0.80,
    "question_quality": 0.85,
    "vtu_compliance": 0.87,
    "strengths": [...],
    "weaknesses": [...],
    "recommendations": [...]
  },
  "all_results": [...]
}
```

## Evaluation Criteria

### Scoring Weights (VTU Standard)
- **Syllabus Coverage**: 30%
- **Bloom's Taxonomy Distribution**: 25%
- **Unit-wise Coverage**: 20%
- **Marks Distribution**: 15%
- **Question Quality**: 10%

### Bloom's Taxonomy Levels
1. Remember (0-15%)
2. Understand (15-25%)
3. Apply (25-35%)
4. Analyze (20-30%)
5. Evaluate (10-15%)
6. Create (5-10%)

## Project Structure

```
ExamVault/
├── backend/
│   ├── scrutiny/
│   │   ├── ai_comparison.py          # Core AI module
│   │   ├── comparison_api.py         # REST API endpoints
│   │   ├── models.py                 # Database models
│   │   ├── serializers.py            # DRF serializers
│   │   ├── views.py                  # Django views
│   │   └── urls.py                   # URL routing
│   ├── requirements.txt              # Python dependencies
│   ├── manage.py                     # Django CLI
│   └── .env                          # Environment variables
├── frontend/                          # React/Vue frontend
└── README.md                          # This file
```

## Environment Variables

```bash
# Required
GEMINI_API_KEY=your_google_gemini_api_key

# Optional
DEBUG=True
DATABASE_URL=postgresql://user:password@localhost/examvault
```

## Performance Metrics

- **Average Analysis Time**: 30-60 seconds per paper
- **Supported File Formats**: PDF, TXT
- **Maximum Syllabus Size**: 50MB
- **Maximum Paper Size**: 20MB

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: support@examvault.dev

## Acknowledgments

- Google Gemini API for AI capabilities
- VTU for examination standards
- Django community for the framework
