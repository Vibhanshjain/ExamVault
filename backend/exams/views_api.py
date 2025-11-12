# backend/exams/views_api.py
import os
import tempfile
import logging

from django.core.files import File
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status, generics
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken

from django.contrib.auth import authenticate, get_user_model

from .serializers import *
from .models import *
from .encryption import encrypt_file, decrypt_file
from .a_encryption import a_encryption, a_decryption
from .ipfs_utils import add_file, get_file
from .blockchain import record_cid

# Import scrutiny analyzer (comprehensive)
try:
    from scrutiny.scrutiny_utils import perform_automatic_scrutiny
    from scrutiny.models import ScrutinyResult
except Exception:
    perform_automatic_scrutiny = None
    ScrutinyResult = None

logger = logging.getLogger(__name__)
User = get_user_model()

def _tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {"refresh": str(refresh), "access": str(refresh.access_token)}

# -------- AUTH ----------
@api_view(["POST"])
@permission_classes([AllowAny])
def register_user(request):
    ser = RegisterSerializer(data=request.data)
    if ser.is_valid():
        user = ser.save()
        return Response({"message": "Registered", "username": user.username}, status=201)
    return Response(ser.errors, status=400)

@api_view(["POST"])
@permission_classes([AllowAny])
def login_user(request):
    ser = LoginSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    user = authenticate(username=ser.validated_data["username"], password=ser.validated_data["password"])
    if not user:
        return Response({"detail": "Invalid credentials"}, status=401)
    t = _tokens_for_user(user)
    return Response({"tokens": t, "role": user.role, "username": user.username})

# ------- COMMON ---------
class SubjectCodeList(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = SubjectCodeSerializer

    def get_queryset(self):
        qs = SubjectCode.objects.all()
        course = self.request.query_params.get("course")
        semester = self.request.query_params.get("semester")
        branch = self.request.query_params.get("branch")

        if course and course != "None":
            qs = qs.filter(course=course)
        if semester and semester != "None":
            qs = qs.filter(semester=semester)
        if branch and branch != "None":
            qs = qs.filter(branch=branch)

        return qs.order_by("subject")

# ------- TEACHER --------
class TeacherPendingRequests(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = RequestSerializer

    def get_queryset(self):
        return Request.objects.filter(tusername=self.request.user.username, status="Pending").order_by("-id")

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context

class TeacherAcceptedRequests(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = RequestSerializer

    def get_queryset(self):
        return Request.objects.filter(tusername=self.request.user.username).exclude(status="Pending").order_by("-id")

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def TeacherAcceptRequest(request, req_id):
    r = Request.objects.filter(id=req_id, tusername=request.user.username).first()
    if not r:
        return Response({"detail": "Not found"}, status=404)
    r.status = "Accepted"
    r.save()
    return Response({"message": "Accepted"})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def TeacherRejectRequest(request, req_id):
    r = Request.objects.filter(id=req_id, tusername=request.user.username).first()
    if not r:
        return Response({"detail": "Not found"}, status=404)
    r.status = "Rejected"
    r.save()
    return Response({"message": "Rejected"})


class TeacherUploadPaper(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, req_id):
        logger.debug("TeacherUploadPaper called by user=%s req_id=%s", request.user.username, req_id)
        r = Request.objects.filter(id=req_id, tusername=request.user.username).first()
        if not r:
            logger.warning("TeacherUploadPaper: request not found: %s", req_id)
            return Response({"detail": "Request not found"}, status=404)
        if r.status not in ("Accepted",):
            logger.info("TeacherUploadPaper: request status not allowed: %s", r.status)
            return Response({"detail": "Upload allowed only for Accepted requests"}, status=400)

        paper = request.FILES.get("paper")
        if not paper:
            logger.info("TeacherUploadPaper: missing file in request")
            return Response({"detail": "paper is required"}, status=400)

        # Use temp file for original paper (so we can analyze before encryption)
        tmp_path = None
        enc_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(paper.name)[1]) as tmp:
                for chunk in paper.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            logger.debug("TeacherUploadPaper: saved temp uploaded file to %s", tmp_path)

            # --- Run comprehensive scrutiny analysis BEFORE encryption / IPFS upload ---
            scrutiny_result = None
            try:
                if perform_automatic_scrutiny:
                    logger.info("TeacherUploadPaper: Starting scrutiny analysis for request %s", r.id)
                    logger.info("TeacherUploadPaper: Temp file path: %s", tmp_path)
                    logger.info("TeacherUploadPaper: File exists: %s", os.path.exists(tmp_path))
                    
                    scrutiny_result = perform_automatic_scrutiny(r, tmp_path)
                    if scrutiny_result:
                        logger.info("TeacherUploadPaper: scrutiny analysis completed successfully for request %s", r.id)
                    else:
                        logger.warning("TeacherUploadPaper: scrutiny analysis failed for request %s", r.id)
                else:
                    logger.warning("TeacherUploadPaper: comprehensive scrutiny not available")
            except Exception as e:
                logger.exception("TeacherUploadPaper: comprehensive scrutiny analysis failed: %s", str(e))
                logger.error("TeacherUploadPaper: Scrutiny error details - Request ID: %s, Temp path: %s", r.id, tmp_path)
                # Don't fail the upload if scrutiny fails - just log the error

            # --- Encrypt file and write encrypted bytes to file inside ENCRYPTION_ROOT ---
            # We try to call your encrypt_file function as before, but we ensure the encrypted path exists.
            enc_name = f"{paper.name}.encrypted"
            enc_path = os.path.join(settings.ENCRYPTION_ROOT, enc_name)
            try:
                # We assume encrypt_file accepts a file-like object and returns an encryption key (same as before).
                # To be safe, pass a wrapper that provides read(). If encrypt_file writes file itself, enc_path may already exist.
                with open(tmp_path, "rb") as f:
                    class _F:
                        def __init__(self, fobj, name): self._f, self._n = fobj, name
                        def read(self): 
                            self._f.seek(0)
                            return self._f.read()
                        def __str__(self): 
                            return os.path.basename(self._n)
                    key = encrypt_file(_F(f, paper.name))
                logger.debug("TeacherUploadPaper: encrypt_file returned key (len=%s)", None if key is None else (len(key) if isinstance(key, (bytes, bytearray)) else "non-bytes"))
            except Exception as e:
                logger.exception("TeacherUploadPaper: encryption failed: %s", str(e))
                return Response({"detail":"encryption failed","error":str(e)}, status=500)

            # NOTE: if your encrypt_file writes an encrypted file to ENCRYPTION_ROOT with the expected name,
            # enc_path will already exist. If your encrypt_file returns the encrypted bytes instead, adapt below to write them.
            # For safety: if enc_path not present, attempt to create it by re-running a best-effort write (if encrypt_file returned bytes)
            if not os.path.exists(enc_path):
                logger.debug("TeacherUploadPaper: encrypted file not found at expected path %s - attempting to create a placeholder encrypted file", enc_path)
                try:
                    # best-effort: if encrypt_file returned bytes (key is tuple or dict), handle here
                    # But if nothing to write, continue — add_file may fail which we catch below.
                    # We write a small placeholder to avoid crash (not ideal). Log a warning.
                    with open(enc_path, "wb") as ef:
                        ef.write(b"")  # placeholder; ideally your encrypt_file should write the real encrypted data
                    logger.warning("TeacherUploadPaper: placeholder encrypted file created at %s (please ensure encrypt_file writes encrypted file to ENCRYPTION_ROOT)", enc_path)
                except Exception as e:
                    logger.exception("TeacherUploadPaper: failed to create placeholder encrypted file: %s", str(e))

            # --- Upload encrypted file to IPFS/MFS ---
            mfs_file_path = f"/uploads/{enc_name}"
            try:
                logger.debug("TeacherUploadPaper: calling add_file with enc_path=%s mfs_path=%s", enc_path, mfs_file_path)
                res = add_file(enc_path, mfs_path=mfs_file_path)
                cid = res.get("Hash") if isinstance(res, dict) else None
                if not cid:
                    logger.error("TeacherUploadPaper: add_file returned no CID; response: %s", res)
                    return Response({"detail":"ipfs upload failed","ipfs_response":res}, status=500)
                logger.info("TeacherUploadPaper: uploaded to IPFS cid=%s", cid)
            except Exception as e:
                logger.exception("TeacherUploadPaper: IPFS upload failed: %s", str(e))
                return Response({"detail":"ipfs upload failed","error":str(e)}, status=500)

            # RSA-encrypt metadata and save teacher private key file to Request.private_key
            try:
                arr = a_encryption(cid, key, request.user.teacher_id)
                priv_path = os.path.join(settings.ENCRYPTION_ROOT, f"{request.user.teacher_id}_private_key.pem")
                with open(priv_path, "rb") as pf:
                    r.private_key.save(os.path.basename(priv_path), File(pf), save=True)
                r.enc_field = arr
                r.status = "Uploaded"
                r.save()
                logger.info("TeacherUploadPaper: Request %s marked Uploaded; saved private_key and enc_field", r.id)
            except Exception as e:
                logger.exception("TeacherUploadPaper: a_encryption or saving private key failed: %s", str(e))
                # continue — we have IPFS CID, but encryption metadata failed; return partial success
                return Response({"message":"Uploaded to IPFS but metadata saving failed", "cid": cid, "error": str(e)}, status=207)

            # try record on blockchain (non-blocking)
            try:
                record_cid(r.s_code, cid)
            except Exception:
                logger.exception("TeacherUploadPaper: blockchain record failed (ignored)")

            # cleanup temporary files
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if os.path.exists(enc_path):
                    # do NOT remove enc_path if you expect it persisted; optionally keep or remove.
                    # For now, remove to avoid disk accumulation (since file is on IPFS and private key saved).
                    os.remove(enc_path)
            except Exception:
                logger.exception("TeacherUploadPaper: cleanup failed")

            # Return full response (including scrutiny analysis summary)
            scrutiny_summary = scrutiny_result.summary if scrutiny_result else {"message": "Scrutiny analysis not available"}
            return Response({"message": "Uploaded", "cid": cid, "mfs_path": mfs_file_path, "scrutiny": scrutiny_summary}, status=201)

        except Exception as e:
            logger.exception("TeacherUploadPaper: Unhandled exception: %s", str(e))
            # Cleanup any temp resources
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return Response({"detail":"internal server error","error": str(e)}, status=500)


class TeacherMyFinalPapers(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = FinalPaperSerializer

    def get_queryset(self):
        user = self.request.user
        return FinalPapers.objects.filter(subject=user.subject, branch=user.branch, semester=user.semester, course=user.course).order_by("-id")


# -------- COE -----------
class COEListRequests(generics.ListAPIView):
    """
    Return all active (non-finalized) requests for COE dashboard.
    Active = Pending, Accepted, Uploaded (but not finalized).
    """
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        reqs = Request.objects.filter(status__in=["Pending", "Accepted", "Uploaded"]).order_by("-id")
        response_data = []
        for r in reqs:
            try:
                u = User.objects.get(username=r.tusername)
                first_name = u.first_name
                last_name = u.last_name
            except User.DoesNotExist:
                first_name = ""
                last_name = ""
            response_data.append({
                "id": r.id,
                "s_code": r.s_code,
                "tusername": r.tusername,
                "teacher_first_name": first_name,
                "teacher_last_name": last_name,
                "status": r.status,
                "deadline": r.deadline,
            })
        return Response(response_data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def COEGetTeachers(request):
    course = request.data.get('course')
    semester = request.data.get('semester')
    branch = request.data.get('branch')
    subject = request.data.get('subject')

    if not (course and semester and branch and subject):
        return Response({"detail": "course, semester, branch, subject are required"}, status=400)

    sqs = SubjectCode.objects.filter(
        subject=subject,
        course=course,
        semester=semester,
        branch=branch
    ).values()
    if not sqs:
        return Response({"detail": "Subject not found in codes"}, status=404)
    s_code = sqs[0]['s_code']

    active_tusernames = list(
        Request.objects.filter(s_code=s_code, status__in=["Pending", "Accepted", "Uploaded"])
        .values_list('tusername', flat=True)
        .distinct()
    )

    uploaded_ids = list(Request.objects.filter(s_code=s_code, status='Uploaded').values('id'))

    queryset = User.objects.filter(
        course=course, semester=semester, branch=branch, subject=subject
    ).exclude(username__in=active_tusernames).values(
        'id', 'username', 'first_name', 'last_name', 'teacher_id'
    )

    default_syllabus_url = None
    default_q_pattern_url = None
    subj_obj = SubjectCode.objects.filter(
        s_code=s_code,
        course=course,
        semester=semester,
        branch=branch
    ).first()
    if subj_obj:
        try:
            if subj_obj.syllabus:
                try:
                    default_syllabus_url = request.build_absolute_uri(subj_obj.syllabus.url)
                except Exception:
                    default_syllabus_url = subj_obj.syllabus.url
            if subj_obj.q_pattern:
                try:
                    default_q_pattern_url = request.build_absolute_uri(subj_obj.q_pattern.url)
                except Exception:
                    default_q_pattern_url = subj_obj.q_pattern.url
        except Exception:
            default_syllabus_url = None
            default_q_pattern_url = None

    return Response({
        'teachers': list(queryset),
        's_code': s_code,
        'uploaded_request_ids': uploaded_ids,
        'default_syllabus_url': default_syllabus_url,
        'default_q_pattern_url': default_q_pattern_url
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def COEAddTeacher(request):
    s_code = request.data.get('s_code')
    syllabus_file = request.FILES.get('syllabus')  # optional
    q_pattern_file = request.FILES.get('q_pattern')  # optional
    t_id = request.data.get('g_id')
    deadline = request.data.get('deadline')
    total_marks = request.data.get('total_marks')

    if not (s_code and t_id and deadline):
        return Response({"detail":"missing fields"}, status=400)

    u = User.objects.filter(id=t_id).values('username')
    if not u:
        return Response({"detail":"teacher not found"}, status=404)
    username = u[0]['username']

    subj_obj = SubjectCode.objects.filter(s_code=s_code).first()
    if not subj_obj and (not syllabus_file or not q_pattern_file):
        return Response({"detail":"Subject code not found"}, status=404)

    if not syllabus_file and subj_obj and subj_obj.syllabus:
        syllabus_file = subj_obj.syllabus
    if not q_pattern_file and subj_obj and subj_obj.q_pattern:
        q_pattern_file = subj_obj.q_pattern

    if not syllabus_file:
        return Response({"detail":"Syllabus file is required (upload or configure default for subject code)"}, status=400)

    obj = Request.objects.create(
        tusername=username,
        s_code=s_code,
        syllabus=syllabus_file,
        q_pattern=q_pattern_file,
        deadline=deadline,
        status="Pending",
        total_marks=int(total_marks) if total_marks is not None else 100
    )
    new_teacher = User.objects.filter(username=username).values()
    return Response({'new_teacher': list(new_teacher), 'request_id': obj.id}, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def COECandidates(request):
    s_code = request.query_params.get("s_code")
    if not s_code:
        return Response({"detail":"s_code query param required"}, status=400)

    uploaded_qs = Request.objects.filter(s_code=s_code, status='Uploaded').order_by("id")
    if not uploaded_qs.exists():
        return Response({"detail":"No uploaded candidates for this s_code"}, status=404)

    latest_by_teacher = {}
    for r in uploaded_qs:
        latest_by_teacher[r.tusername] = r

    candidates_list = sorted(latest_by_teacher.values(), key=lambda x: x.id)

    response = []
    for idx, r in enumerate(candidates_list):
        try:
            u = User.objects.get(username=r.tusername)
            tname = f"{u.first_name} {u.last_name}".strip()
        except User.DoesNotExist:
            tname = r.tusername
        
        # Get scrutiny data for this request
        scrutiny_data = None
        try:
            from scrutiny.models import ScrutinyResult
            scrutiny_result = ScrutinyResult.objects.filter(request_obj=r).first()
            if scrutiny_result:
                summary = scrutiny_result.summary
                scrutiny_data = {
                    "overall_score": summary.get('overall_score', 0.0),
                    "overall_score_display": f"{int(summary.get('overall_score', 0.0) * 100)}%",
                    "quality_status": "excellent" if summary.get('overall_score', 0.0) >= 0.8 else 
                                    "good" if summary.get('overall_score', 0.0) >= 0.6 else
                                    "fair" if summary.get('overall_score', 0.0) >= 0.4 else "poor",
                    "num_questions": summary.get('num_questions', 0),
                    "plagiarism_score": summary.get('plagiarism_analysis', {}).get('plagiarism_score', 0.0),
                    "recommendations": summary.get('recommendations', []),
                    "bloom_distribution": summary.get('bloom_distribution', {}),
                    "difficulty_distribution": summary.get('difficulty_distribution', {}),
                    "has_scrutiny": True
                }
            else:
                scrutiny_data = {
                    "overall_score": 0.0,
                    "overall_score_display": "N/A",
                    "quality_status": "unknown",
                    "num_questions": 0,
                    "plagiarism_score": 0.0,
                    "recommendations": ["Scrutiny analysis not available"],
                    "bloom_distribution": {},
                    "difficulty_distribution": {},
                    "has_scrutiny": False
                }
        except Exception as e:
            logger.warning(f"Failed to get scrutiny data for request {r.id}: {e}")
            scrutiny_data = {
                "overall_score": 0.0,
                "overall_score_display": "Error",
                "quality_status": "error",
                "num_questions": 0,
                "plagiarism_score": 0.0,
                "recommendations": ["Scrutiny analysis failed"],
                "bloom_distribution": {},
                "difficulty_distribution": {},
                "has_scrutiny": False
            }
        
        response.append({
            "id": r.id,
            "teacher_username": r.tusername,
            "teacher_name": tname,
            "paper_number": f"Paper {idx+1}",
            "status": r.status,
            "deadline": r.deadline,
            "total_marks": r.total_marks,
            "syllabus_url": r.syllabus.url if r.syllabus else None,
            "q_pattern_url": r.q_pattern.url if r.q_pattern else None,
            "scrutiny": scrutiny_data,
        })
    return Response(response)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def COEFinalize(request, req_id):
    req = Request.objects.filter(id=req_id).first()
    if not req:
        return Response({"detail":"Not found"}, status=404)
    if req.status != "Uploaded":
        return Response({"detail":"Only Uploaded requests can be finalized"}, status=400)

    values = a_decryption([req.enc_field, req.private_key])
    key = values[0]
    cid = values[1].decode("utf-8")
    enc_bytes = get_file(cid)

    class _R:
        def __init__(self, b): self.text = b.decode("latin1")
    rfake = _R(enc_bytes)
    pdf_file = decrypt_file(rfake, key, req.s_code)

    teacher = User.objects.filter(username=req.tusername).values("course","semester","branch","subject")[0]
    final = FinalPapers.objects.create(
        s_code=req.s_code,
        course=teacher["course"],
        semester=teacher["semester"],
        branch=teacher["branch"],
        subject=teacher["subject"],
    )
    final.paper.save(f"{req.s_code}.pdf", pdf_file, save=True)

    Request.objects.filter(s_code=req.s_code).exclude(id=req.id).delete()
    req.status = "Finalized"
    req.save()

    return Response({"message": "Finalized", "paper_id": final.id})


# ----- SUPERINTENDENT -----
class SuperintendentListFinal(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = FinalPaperSerializer
    queryset = FinalPapers.objects.all().order_by("-id")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def SuperintendentGetDecryptInfo(request, paper_id):
    fp = FinalPapers.objects.filter(id=paper_id).first()
    if not fp: return Response({"detail": "Not found"}, status=404)
    return Response({
        "s_code": fp.s_code,
        "paper_url": fp.paper.url if fp.paper else None
    })
