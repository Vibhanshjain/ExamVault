from django.db import models
from django.contrib.auth.models import AbstractUser
import datetime
from django.contrib.postgres.fields import ArrayField

def teacherID():
    t_id = 'TEA-1'
    try:
        prev = CustomUser.objects.values('teacher_id').last()
        prev = prev['teacher_id']
        number = int(prev.split('-')[1]) + 1
        t_id = 'TEA-' + str(number)
    except:
        t_id = 'TEA-1'
    return t_id

ROLE = (
    ('teacher', 'teacher'),
    ('coe', 'coe'),
    ('superintendent', 'superintendent')
)

COURSE = (
    ('None', 'None'),
    ('B.E.', 'B.E.'),
    ('M.E.', 'M.E.')
)

SEM = (
    ('None', 'None'),
    ('I', 'I'),
    ('II', 'II'),
    ('III', 'III'),
    ('IV', 'IV'),
    ('V', 'V'),
    ('VI', 'VI'),
    ('VII', 'VII'),
    ('VIII', 'VIII')
)

BRANCH = (
    ('None', 'None'),
    ('CSE', 'CSE'),
    ('IT', 'IT'),
    ('ECE', 'ECE'),
    ('EEE', 'EEE'),
    ('MECH', 'MECH'),
    ('BioTech', 'BioTech')
)

SUB = (
    ('None', 'None'),
    ('Internet of Things', 'Internet of Things'),
    ('Parallel Computing', 'Parallel Computing'),
    ('Cryptography', 'Cryptography'),
    ('Big Data Analytics', 'Big Data Analytics'),
    ('Cloud Computing', 'Cloud Computing')
)

STATUS = (
    ('Pending', 'Pending'),
    ('Accepted', 'Accepted'),
    ('Uploaded', 'Uploaded'),
    ('Finalized', 'Finalized'),
    ('Rejected', 'Rejected'),
)


class CustomUser(AbstractUser):
    teacher_id = models.CharField(max_length=20, default=teacherID, blank=True)
    course = models.CharField(max_length=4, choices=COURSE, default='None')
    semester = models.CharField(max_length=4, choices=SEM, default='None')
    branch = models.CharField(max_length=40, choices=BRANCH, default='None')
    subject = models.CharField(max_length=30, choices=SUB, default='None')
    role = models.CharField(max_length=20, choices=ROLE, default='teacher')

    def __str__(self):
        return self.username


class Request(models.Model):
    tusername = models.CharField(max_length=40, default='None')
    s_code = models.CharField(max_length=7, default="None")
    syllabus = models.FileField(upload_to='syllabus/', null=True, blank=True)
    q_pattern = models.FileField(upload_to='q_patterns/', null=True, blank=True)
    deadline = models.DateField(default=datetime.date.today)
    status = models.CharField(max_length=10, default='Pending', choices=STATUS)
    enc_field = ArrayField(models.BinaryField(max_length=500, default=None), default=list, blank=True)
    private_key = models.FileField(upload_to='private_keys/', null=True, blank=True)
    total_marks = models.IntegerField(default=100)

    def __str__(self):
        return f"{self.tusername} - {self.s_code}"


class FinalPapers(models.Model):
    s_code = models.CharField(max_length=7, default="None")
    course = models.CharField(max_length=4, default='None')
    semester = models.CharField(max_length=4, default='None')
    branch = models.CharField(max_length=40, default='None')
    subject = models.CharField(max_length=30, default='None')
    paper = models.FileField(upload_to='final_papers/', null=True, blank=True)

    def __str__(self):
        return self.s_code


class SubjectCode(models.Model):
    s_code = models.CharField(max_length=7)
    subject = models.CharField(max_length=40)
    course = models.CharField(max_length=4, choices=COURSE, default='None')
    semester = models.CharField(max_length=4, choices=SEM, default='None')
    branch = models.CharField(max_length=40, choices=BRANCH, default='None')
    syllabus = models.FileField(upload_to='subject_syllabus/', null=True, blank=True)
    q_pattern = models.FileField(upload_to='subject_qpatterns/', null=True, blank=True)

    def __str__(self):
        return self.subject
