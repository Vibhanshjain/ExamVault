"""
Django REST API endpoints for exam paper comparison.
Provides endpoints to compare and rank exam papers using AI.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.conf import settings
import os
import logging
from .ai_comparison import compare_multiple_papers, analyze_single_paper, export_comparison_results

logger = logging.getLogger(__name__)


class ComparisonViewSet(viewsets.ViewSet):
    """
    ViewSet for exam paper comparison operations.
    Endpoints:
    - POST /compare/ : Compare multiple papers
    - POST /analyze/ : Analyze a single paper
    - POST /rank/ : Rank papers and get best paper
    """
    
    @action(detail=False, methods=['post'])
    def compare(self, request):
        """
        Compare multiple exam papers and rank them.
        
        Expected request data:
        {
            'syllabus_path': 'path/to/syllabus.pdf',
            'papers': [
                {'path': 'path/to/paper1.pdf', 'name': 'Paper 2024'},
                {'path': 'path/to/paper2.pdf', 'name': 'Paper 2023'}
            ],
            'total_marks': 100
        }
        """
        try:
            syllabus_path = request.data.get('syllabus_path')
            papers = request.data.get('papers', [])
            total_marks = request.data.get('total_marks', 100)
            
            if not papers:
                return Response(
                    {'error': 'No papers provided for comparison'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Convert to tuples for comparison function
            paper_tuples = [(p['path'], p['name']) for p in papers]
            
            # Run comparison
            results = compare_multiple_papers(syllabus_path, paper_tuples, total_marks)
            
            # Format results for response
            formatted_results = [
                {
                    'name': r.paper_name,
                    'overall_score': r.overall_score,
                    'syllabus_coverage': r.syllabus_coverage,
                    'bloom_distribution': r.bloom_distribution,
                    'unit_coverage': r.unit_coverage,
                    'marks_distribution': r.marks_distribution,
                    'question_quality': r.question_quality,
                    'vtu_compliance': r.vtu_compliance,
                    'strengths': r.strengths,
                    'weaknesses': r.weaknesses,
                    'recommendations': r.recommendations
                }
                for r in results
            ]
            
            return Response({
                'status': 'success',
                'papers_analyzed': len(results),
                'best_paper': formatted_results[0] if formatted_results else None,
                'all_results': formatted_results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Comparison error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'])
    def analyze(self, request):
        """
        Analyze a single exam paper.
        
        Expected request data:
        {
            'syllabus_path': 'path/to/syllabus.pdf',
            'paper_path': 'path/to/paper.pdf',
            'paper_name': 'Paper Name',
            'total_marks': 100
        }
        """
        try:
            syllabus_path = request.data.get('syllabus_path')
            paper_path = request.data.get('paper_path')
            paper_name = request.data.get('paper_name', 'Exam Paper')
            total_marks = request.data.get('total_marks', 100)
            
            if not paper_path:
                return Response(
                    {'error': 'Paper path is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = analyze_single_paper(syllabus_path, paper_path, paper_name, total_marks)
            
            return Response({
                'status': 'success',
                'analysis': {
                    'name': result.paper_name,
                    'overall_score': result.overall_score,
                    'syllabus_coverage': result.syllabus_coverage,
                    'bloom_distribution': result.bloom_distribution,
                    'unit_coverage': result.unit_coverage,
                    'marks_distribution': result.marks_distribution,
                    'question_quality': result.question_quality,
                    'vtu_compliance': result.vtu_compliance,
                    'strengths': result.strengths,
                    'weaknesses': result.weaknesses,
                    'recommendations': result.recommendations
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
