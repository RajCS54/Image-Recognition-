from re import A
import cv2 as cv
import os

file = str(input("Enter the file name : "))

img = cv.imread(f'Photos/{file}.jpg')

img_size = os.path.getsize(f'Photos/{file}.jpg')

thres = 0.6
size = 1000

if len(img) > size:
    pic = cv.resize(img, (800, 600))
else:
    pic = img


configure_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

class_datasets = []
data_set = 'datasets.names'
with open(data_set, 'rt') as r:
    class_datasets = r.read().split('\n')
# print(class_datasets)


net = cv.dnn_DetectionModel(frozen_model, configure_file)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net .setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)  # converts bgr 2 rgb

class_ID, confidence, boundary_box = net.detect(pic, confThreshold=thres)
zipped = zip(class_ID.flatten(), confidence.flatten(), boundary_box)
for classid, confident, box in zipped:
    cv.rectangle(pic, box, color=(440, 0, 0), thickness=1)  # bgr
    cv.putText(pic, class_datasets[classid-1].upper(), (box[0]+5, box[1]+30),
               cv.FONT_HERSHEY_TRIPLEX, 1, (0, 600, 0), 1)

cv.imshow('Image Recognition', pic)
cv.waitKey(0)
