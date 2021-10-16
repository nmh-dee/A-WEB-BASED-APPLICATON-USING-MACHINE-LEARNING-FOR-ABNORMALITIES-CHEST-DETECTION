from django.contrib import admin
from .models import PPatient,PPatientImages, FVC
# Register your models here.

admin.site.register(PPatientImages)
admin.site.register(PPatient)
admin.site.register(FVC)