import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw


def img_cvread(path):
    img = cv2.imread(path)
    return img


def drawRectBox(image, rect, addText, fontC):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 0, 255), 4,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]), int(rect[1] - 70)), addText.encode("utf-8").decode("utf-8"), (160, 32, 240),
              font=fontC)
    imagex = np.array(img)
    return imagex
