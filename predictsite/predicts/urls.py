from django.urls import path, reverse_lazy
from . import views

app_name='predicts'
urlpatterns= [
    path('',views.PredictListView.as_view(), name='all'),
    path('predict/<int:pk>', views.PredictDetailView.as_view(), name='predict_detail'),
    path('xray/<int:pk>', views.stream_file, name='xray'),
    #path('xray_predicted/<int:pk>', views.stream_after_file, name='xray_predicted'),

    #path('predict/<int:pk>/verification',views.VerificationCreateView.as_view(), name='predict_verification_create'),
    #path('verification/<int:pk>/delete', views.VerificationDeleteView.as_view(success_url=reverse_lazy('predicts:all')), name='predict_verification_delete'),

    path('predict/create',views.PredictCreateView.as_view(), name='predict_create'),

    path('predict/<int:pk>/update',views.PredictUpdateView.as_view(success_url=reverse_lazy('predicts:all')), name='predict_update'),

    path('predict/<int:pk>/delete',views.PredictDeleteView.as_view(success_url=reverse_lazy('predicts:all')), name='predict_delete'),


    ]