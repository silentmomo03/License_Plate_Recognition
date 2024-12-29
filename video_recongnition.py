#coding:utf-8
import cv2
from ultralytics import YOLO
import detect_tools as tools
from PIL import ImageFont
from paddleocr import PaddleOCR
def get_license_result(ocr,image):
    """
    image:输入的车牌截取照片
    输出，车牌号与置信度
    """
    result = ocr.ocr(image, cls=True)[0]
    if result:
        license_name, conf = result[0][1]
        if '·' in license_name:
            license_name = license_name.replace('·', '')
        return license_name, conf
    else:
        return None, None
fontC = ImageFont.truetype("Font/platech.ttf", 50, 0)
# 加载ocr模型
cls_model_dir = 'paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
rec_model_dir = 'paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
ocr = PaddleOCR(use_angle_cls=False, lang="ch", det=False, cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir)
# 所需加载的模型目录
path = 'runs/detect/train2/weights/best.pt'
# 加载预训练模型
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# 需要检测的图片地址
video_path = "TestFiles/1.mp4"
cap = cv2.VideoCapture(video_path)
# 获取原视频的宽度、高度和帧率
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (640, 360)
# 创建 VideoWriter 对象，用于保存灰度视频
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 其中*'MP4V'和 'M', 'P', '4', 'V'等效
video = cv2.VideoWriter('output.mp4', fourcc, fps, size)  # size可能会与result尺寸不匹配
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv11 inference on the frame
        results = model(frame)[0]
        location_list = results.boxes.xyxy.tolist()
        if len(location_list) >= 1:
            location_list = [list(map(int, e)) for e in location_list]
            # 截取每个车牌区域的照片
            license_imgs = []
            for each in location_list:
                x1, y1, x2, y2 = each
                cropImg = frame[y1:y2, x1:x2]
                license_imgs.append(cropImg)
            # 车牌识别结果
            lisence_res = []
            conf_list = []
            for each in license_imgs:
                license_num, conf = get_license_result(ocr, each)
                if license_num:
                    lisence_res.append(license_num)
                    conf_list.append(conf)
                else:
                    lisence_res.append('无法识别')
                    conf_list.append(0)
            for text, box in zip(lisence_res, location_list):
                frame = tools.drawRectBox(frame, box, text, fontC)
        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        print(frame.shape)
        # 保存灰度帧到输出视频
        video.write(frame)
        cv2.imshow("YOLOv11 Detection", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()