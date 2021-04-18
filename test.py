
import cv2
import argparse
import numpy as np
import imutils
from django.conf import settings

image = cv2.imread(args.image)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392 
def prepare(image):
    model_weights = "yolov3_best.weights"
    model_cfg = "yolov3.cfg"
    with open(os.path.join(settings.MODEL_ROOT, model_classes)) as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(os.path.join(settings.MODEL_ROOT, model_weights),
                                         os.path.join(settings.MODEL_ROOT, model_cfg))

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False) 
    net.setInput(blob)
    return net
 
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    model_classes = "yolo.names"
    with open(os.path.join(settings.MODEL_ROOT, model_classes)) as f:
        classes = [line.strip() for line in f.readlines()]
    label = str(classes[class_id])
    cv2.putText(img, label, (x - 20, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
    return img 
    
def image_detect(image,net):
    outs = net.forward(get_output_layers(net)) 
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
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    image= imutils.resize(image, width=500) 
    return image

    cv2.imshow("object detection", image)