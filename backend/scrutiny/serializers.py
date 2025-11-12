from rest_framework import serializers
from .models import ScrutinyResult

class ScrutinyResultSerializer(serializers.ModelSerializer):
    request_info = serializers.SerializerMethodField()
    overall_score_display = serializers.SerializerMethodField()
    quality_status = serializers.SerializerMethodField()
    
    class Meta:
        model = ScrutinyResult
        fields = [
            "id", 
            "request_obj", 
            "request_info",
            "summary", 
            "created_at",
            "overall_score_display",
            "quality_status"
        ]
    
    def get_request_info(self, obj):
        """Include basic request information for easier frontend display"""
        if obj.request_obj:
            return {
                "id": obj.request_obj.id,
                "subject_code": obj.request_obj.s_code,  # Fixed: use s_code instead of subject_code
                "teacher_name": obj.request_obj.tusername,  # Fixed: use tusername field
                "status": obj.request_obj.status,
                "created_at": obj.request_obj.created_at if hasattr(obj.request_obj, 'created_at') else None
            }
        return None
    
    def get_overall_score_display(self, obj):
        """Convert numeric score to percentage display"""
        score = obj.summary.get('combined_score', obj.summary.get('overall_score', 0.0))
        return f"{int(score * 100)}%"
    
    def get_quality_status(self, obj):
        """Determine quality status based on score"""
        score = obj.summary.get('combined_score', obj.summary.get('overall_score', 0.0))
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
