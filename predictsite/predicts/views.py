from django.shortcuts import render, redirect, get_object_or_404
from .models import Predict, Verification
from .owner import OwnerListView, OwnerDetailView,OwnerCreateView, OwnerUpdateView, OwnerDeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy, reverse
from django.views import View
from .forms import CreateForm, VerifyForm
from django.http import HttpResponse

import cv2
import numpy as np

import os
from django.conf import settings
from PIL import Image
   


# Create your views here.

class PredictListView(OwnerListView):
    model = Predict
    template_name = 'predicts/predict_list.html'
class PredictDetailView(OwnerDetailView):
    model = Predict
    template_name = 'predicts/predict_detail.html'

class PredictDeleteView(OwnerDeleteView):
    model = Predict

class PredictCreateView(LoginRequiredMixin, View):
    template_name = 'predicts/predict_form.html'
    success_predict= 'predicts/predict_detail.html'
    #success_url = reverse_lazy('predicts/predict_detail.html')
    def get(self,request,pk = None):
        form = CreateForm()
        ctx = {'form':form}
        return render(request,self.template_name, ctx)
    def post(self,request, pk= None):
        form = CreateForm(request.POST, request.FILES or None)
        if not form.is_valid():
            ctx ={'form':form}
            return render(request,self.template_name,ctx)
        predict = form.save(commit= True)
        predict.owner = self.request.user
        image =cv2.imread(predict.xray.path)
        net= self.prepare(image)
        img= self.image_detect(image,net)

        print(predict.xray.name.replace("xray","xray_predicted"))
        predict.xray_predicted = predict.xray.name.replace("xray","xray_predicted")
        #predicts/predict/xray_predicted/ewyhccg.png
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT,predict.xray_predicted),img)
        predict.save()
        ctx ={'predict':predict}
        return render(request,self.success_predict,ctx)
    
    def prepare(self,image):
        scale = 0.00392 
        model_weights = "yolov3_best.weights"
        model_cfg = "yolov3.cfg"
        
        net = cv2.dnn.readNet(os.path.join(settings.MODEL_ROOT, model_weights),
                                            os.path.join(settings.MODEL_ROOT, model_cfg))
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False) 
        net.setInput(blob)
        return net
    def get_output_layers(self,net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        model_classes = "yolo.names"
        with open(os.path.join(settings.MODEL_ROOT, model_classes)) as f:
            classes = [line.strip() for line in f.readlines()]
        label = str(classes[class_id])
        cv2.putText(img, label, (x - 20, y - 35), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 1)
        return img 
    def image_detect(self,image,net):
        outs = net.forward(self.get_output_layers(net)) 
        Width = image.shape[0]
        Height = image.shape[1]
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4 
        # Thực hiện xác định bằng HOG và SVM
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h]) 
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold) 
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
            print(x,";",y,";")
        return image
'''
def stream_after_file(request,pk):
    predict = get_object_or_404(Predict,id= pk)
    response = HttpResponse()
    response['Content-Type'] = predict.content_type
    response['Content-Length'] = 
    response.write(cv2.imshow(predict.xray_predicted))
    return response
'''
class PredictUpdateView(LoginRequiredMixin,View):
    template_name = 'predicts/predict_form.html'
    success_url = reverse_lazy('predicts:all')
    def get(self,request,pk):
        predict = get_object_or_404(Predict, id= pk, owner= self.request.user)
        form = CreateForm(instance= predict)
        ctx = {'form':form}
        return render(request, self.template_name, ctx)
    def post(self, request,pk):
        predict= get_object_or_404(Predict,id= pk, owner = self.request.user)
        form = CreateForm (request.POST, request.FILES or None, instance = predict)

        if not form.is_valid():
            ctx = {'form':form}
            return render(request, self.template_name, ctx)
        predict = form.save(commit= False)
        predict.save()
        return redirect(self.success_url)
def stream_file(request,pk):
    predict = get_object_or_404(Predict,id= pk)
    response = HttpResponse()
    response['Content-Type'] = predict.content_type
    response['Content-Length'] = len(predict.xray)
    response.write(predict.xray)
    return response
'''
class VerificationCreateView(LoginRequiredMixin, View):
    def post(self, request, pk):
        a = get_object_or_404(Predict, id = pk)
        verification= Verification(verification= request.POST['verifications'], owner=request.user, predict=a)
        verification.save()
        return redirect(reverse('predicts:predict_detail'), args=[pk])
'''