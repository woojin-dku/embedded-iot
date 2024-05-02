import torch
import numpy as np
from ultralytics import YOLO
import cv2
import pafy

model = YOLO("yolov8n.pt")

# url = 'https://www.youtube.com/watch?v=sD9gTAFDq40'
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
# cap = cv2.VideoCapture(best.url)


# cap = cv2.VideoCapture("dogs.mp4")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    ### Use result!!

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()

