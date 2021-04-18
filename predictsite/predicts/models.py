from django.db import models

# Create your models here.
from django.db import models
from django.core.validators import MinLengthValidator
from django.conf import settings
# Create your models here.
from django.db import models
class Predict(models.Model):
    name_patient = models.CharField(max_length=200,validators=[MinLengthValidator(2, "Name must be greater than 2 characters")]
    )
    #age= models.DecimalField(max_digits=None, decimal_places=)
    #smoke = models.CharField(max_length=50, null = True)
    #gender = models.CharField(max_length=30,null= True)
    #fvc = models.CharField(max_length=10,null = True)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL,default=1,on_delete= models.CASCADE)
    xray = models.ImageField(null= True,upload_to='xray/', editable=True)
    xray_truth = models.ImageField(null = True,blank = True, upload_to= 'truth_xray/' ,editable= True)
    xray_predicted = models.CharField(max_length=200, null= True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    #ct_image = models.BinaryField(null = True, editable= True)
    content_type = models.CharField(max_length= 256, null= True, help_text= 'The MIMEType of the xray file')
    verifications = models.ManyToManyField(settings.AUTH_USER_MODEL ,through= 'Verification', related_name= 'verification_owned')
    def __str__(self):
        return self.name_patient



class Verification(models.Model):
    verification =  models.TextField( validators=[MinLengthValidator(3, "Comment must be greater than 3 characters")]
    )
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete= models.CASCADE)
    predict = models.ForeignKey('Predict', on_delete=models.CASCADE)
    def __str__(self):
        return self.verification