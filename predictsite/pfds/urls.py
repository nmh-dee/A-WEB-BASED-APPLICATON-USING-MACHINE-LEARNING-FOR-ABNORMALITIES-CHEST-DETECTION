from django.urls import path, reverse_lazy
from . import views

app_name = 'pfds'
urlpatterns = [
    path('', views.PListView.as_view(), name = 'all'),
    path('pfd/create', views.MultipleUpload, name= 'pfd_create'),
    path('multipleupload_save', views.Multipleupload_save ),
    path('pfd/<int:pk>', views.chart, name ='pfd_detail'),
    path('pfd/<int:pk>/delete', views.PDeleteView.as_view(), name ='pfd_delete')


]