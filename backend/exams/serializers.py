from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Request, FinalPapers, SubjectCode

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = [
            "username",
            "password",
            "email",
            "first_name",
            "last_name",
            "course",
            "semester",
            "branch",
            "subject",
            "role",
        ]

    def create(self, validated_data):
        pwd = validated_data.pop("password")
        user = User(**validated_data)
        user.set_password(pwd)
        user.save()
        return user

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

class SubjectCodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectCode
        fields = ["id", "s_code", "subject", "course", "semester", "branch"]

class RequestSerializer(serializers.ModelSerializer):
    subject_code = serializers.CharField(source="s_code", read_only=True)

    syllabus_url = serializers.SerializerMethodField()
    q_pattern_url = serializers.SerializerMethodField()

    course = serializers.SerializerMethodField()
    semester = serializers.SerializerMethodField()
    branch = serializers.SerializerMethodField()
    subject = serializers.SerializerMethodField()

    class Meta:
        model = Request
        fields = [
            "id",
            "tusername",
            "subject_code",
            "course",
            "semester",
            "branch",
            "subject",
            "total_marks",
            "deadline",
            "status",
            "syllabus_url",
            "q_pattern_url",
        ]

    def _get_request(self):
        return self.context.get("request", None)

    def get_syllabus_url(self, obj):
        if obj.syllabus:
            try:
                req = self._get_request()
                url = obj.syllabus.url
                return req.build_absolute_uri(url) if req else url
            except Exception:
                return None
        return None

    def get_q_pattern_url(self, obj):
        if obj.q_pattern:
            try:
                req = self._get_request()
                url = obj.q_pattern.url
                return req.build_absolute_uri(url) if req else url
            except Exception:
                return None
        return None

    def _teacher_field(self, obj, field):
        try:
            user = User.objects.get(username=obj.tusername)
            return getattr(user, field, None)
        except User.DoesNotExist:
            return None

    def get_course(self, obj):
        return self._teacher_field(obj, "course")

    def get_semester(self, obj):
        return self._teacher_field(obj, "semester")

    def get_branch(self, obj):
        return self._teacher_field(obj, "branch")

    def get_subject(self, obj):
        return self._teacher_field(obj, "subject")

class FinalPaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = FinalPapers
        fields = ["id", "s_code", "course", "semester", "branch", "subject", "paper"]
