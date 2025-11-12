
from django.urls import path
from . import views_api as v

urlpatterns = [
    
    path("register/", v.register_user),
    path("login/", v.login_user),

    
    path("subject-codes/", v.SubjectCodeList.as_view()),

    
    path("teacher/requests/pending/", v.TeacherPendingRequests.as_view()),
    path("teacher/requests/accepted/", v.TeacherAcceptedRequests.as_view()),
    path("teacher/requests/<int:req_id>/accept/", v.TeacherAcceptRequest),
    path("teacher/requests/<int:req_id>/reject/", v.TeacherRejectRequest),
    path("teacher/requests/<int:req_id>/upload/", v.TeacherUploadPaper.as_view()),
    path("teacher/final-papers/", v.TeacherMyFinalPapers.as_view()),

  
    path("coe/requests/", v.COEListRequests.as_view()),           # list active requests for COE dashboard
    path("coe/teachers/", v.COEGetTeachers),                     # search teachers (returns default files if configured)
    path("coe/requests/add/", v.COEAddTeacher),                  # create request (uses subject defaults if files not sent)
    path("coe/candidates/", v.COECandidates),                    # GET ?s_code=...
    path("coe/requests/<int:req_id>/finalize/", v.COEFinalize),  # finalize chosen candidate

    path("sup/final-papers/", v.SuperintendentListFinal.as_view()),
    path("sup/final-papers/<int:paper_id>/decrypt-info/", v.SuperintendentGetDecryptInfo),
]
