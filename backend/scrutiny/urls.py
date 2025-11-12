from django.urls import path
from . import views

urlpatterns = [
    path("analyze-file/", views.AnalyzeFileAPIView.as_view(), name="scrutiny-analyze-file"),
    path("results/", views.ScrutinyResultsAPIView.as_view(), name="scrutiny-results"),
    path("summary/", views.ScrutinySummaryAPIView.as_view(), name="scrutiny-summary"),
    path("detail/<int:request_id>/", views.ScrutinyDetailAPIView.as_view(), name="scrutiny-detail"),
]
