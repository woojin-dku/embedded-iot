import cv2
from ultralytics import YOLO
import numpy as np
import torch
import pafy


def use_result(results, frame) :
    if (results and results[0]) :
        # print("YOLOv8 Result = ", results[0].__dict__)
        # print("YOLOv8 Result.boxes = ", results[0].boxes)
        # print("YOLOv8 Result.boxes.xyxy = ", results[0].boxes.xyxy)
        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        # print("YOLOv8 Result.boxes.xyxy.cpu() = ", bboxes)
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        names = results[0].names
        # print("Class names = ", names)
        pred_box = zip(classes, bboxes)
        for cls, bbox in pred_box :
            (x, y, x2, y2) = bbox
            # print("bounding box (",x,y,x2,y2,") has class ", cls)
            print("bounding box (",x,y,x2,y2,") has class ", cls, " which is ", names[cls])
            # Draw bounding box for dog
            cv2.rectangle(frame, (x,y), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, str(names[cls]), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

            scale_percent = 60 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
        
            # resize image
            frame_s = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("Img", frame_s)

    return


model = YOLO("best.pt")

url = 'https://www.youtube.com/watch?v=D3eWEvlTGug'
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

# cap = cv2.VideoCapture("dogs.mp4")

# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # For Apple Silicon
    if torch.backends.mps.is_available() :
        results = model(frame, device="mps") # Use MPS
    else :
        results = model(frame)
        use_result(results, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()