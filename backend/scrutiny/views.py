from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
import logging
from .nlp_utils import analyze_file
from .scrutiny_utils import get_scrutiny_summary_for_dashboard
from .models import ScrutinyResult
from .serializers import ScrutinyResultSerializer
from django.shortcuts import get_object_or_404

logger = logging.getLogger(__name__)

class AnalyzeFileAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return Response({"detail":"file is required"}, status=400)
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
            for chunk in f.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        try:
            res = analyze_file(tmp_path)
            return Response(res)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

class ScrutinyResultsAPIView(APIView):
    """
    API endpoint to retrieve scrutiny results for all uploaded papers.
    Used by COE dashboard.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            # Get all scrutiny results with related request information
            results = ScrutinyResult.objects.select_related('request_obj').all().order_by('-created_at')
            
            # Serialize the results
            serializer = ScrutinyResultSerializer(results, many=True)
            
            return Response({
                "results": serializer.data,
                "count": results.count()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Error retrieving scrutiny results: {e}")
            return Response(
                {"detail": "Failed to retrieve scrutiny results"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ScrutinySummaryAPIView(APIView):
    """
    API endpoint to get summary statistics for COE dashboard.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            summary = get_scrutiny_summary_for_dashboard()
            return Response(summary, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Error getting scrutiny summary: {e}")
            return Response(
                {"detail": "Failed to get scrutiny summary"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ScrutinyDetailAPIView(APIView):
    """
    API endpoint to get detailed scrutiny result for a specific paper.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, request_id):
        try:
            scrutiny_result = get_object_or_404(ScrutinyResult, request_obj_id=request_id)
            serializer = ScrutinyResultSerializer(scrutiny_result)
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Error retrieving scrutiny detail for request {request_id}: {e}")
            return Response(
                {"detail": "Failed to retrieve scrutiny details"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
