#coding:utf-8
from ultralytics import YOLO
import cv2
# 所需加载的模型目录
path = 'runs/detect/train2/weights/best.pt'
# 需要检测的图片地址
img_path = "TestFiles/04739583333333333-97_262-214&447_550&589-543&589_215&541_214&447_550&494-0_0_3_17_24_33_24_26-106-220.jpg"
# 加载预训练模型
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# model = YOLO(path, task='detect',conf=0.5)
# 检测图片
results = model(img_path)
res = results[0].plot()
# res = cv2.resize(res,dsize=None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
cv2.imshow("YOLOv11 Detection", res)
cv2.waitKey(0)