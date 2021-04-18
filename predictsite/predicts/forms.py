from django import forms
from .models import Predict

from django.core.files.uploadedfile import InMemoryUploadedFile

from .humanize import naturalsize

from django.core.exceptions import ValidationError
from django.core import validators

class CreateForm(forms.ModelForm):
    max_upload_limit = 2*2048 *2048
    max_upload_limit_text = naturalsize(max_upload_limit)

    xray = forms.FileField(required= False, label = 'File to Upload<='+max_upload_limit_text)
    upload_field_name = 'xray'
    class Meta:
        model = Predict
        fields = ['name_patient','xray']   #,'age'
    def clean(self):
        cleaned_data = super().clean()
        xray = cleaned_data.get('xray')
        if xray is None:
            return
        if len(xray) > self.max_upload_limit:
            self.add_error('xray', "File must be < "+self.max_upload_limit_text+" bytes")
    def save(self,commit= True):
        instance = super(CreateForm,self).save(commit= False)

        # We only need to adjust picture if it is a freshly uploaded file
        f= instance.xray #make a copy
        if isinstance(f,InMemoryUploadedFile):   #Extract data from the form to the model
            bytearr = f.read()
            instance.content_type = f.content_type
            instance.xray = bytearr
        if commit:
            instance.save()
        return instance

# strip means to remove whitespace from the beginning and the end before storing the column
class VerifyForm(forms.Form):
    verification = forms.CharField(required= True,max_length=500, min_length=3, strip=True )