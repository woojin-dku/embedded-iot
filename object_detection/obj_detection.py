import cv2
from ultralytics import YOLO
import numpy as np


cap = cv2.VideoCapture("dogs.mp4")

# Test for capture
# ret, frame = cap.read()
# cv2.imshow("Img", frame)
# cv2.waitKey(0)

model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    #print(results)
    result = results[0]
    #print(result)
    #print(result.boxes)
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    #print(result.boxes.xyxy)
    #print(bboxes)
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        # print("bounding box (",x,y,x2,y2,")")
        # Draw bounding box
        cv2.rectangle(frame, (x,y), (x2,y2), (0,0,255), 2)
        # Display class of bounding box
        # cv2.putText(frame, str(cls), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        cv2.putText(frame, str(result.names[cls]), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    frame_s = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Img", frame_s)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()