import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("vids/FootballVid.mp4")

target_color = None
tolerance = 30
paused = False
state = {"frame": None, "boxes": None}

def on_click(event, x, y, flags, param):
    global target_color
    if event == cv2.EVENT_LBUTTONDOWN and state["frame"] is not None:
        hsv = cv2.cvtColor(state["frame"], cv2.COLOR_BGR2HSV)
        target_color = hsv[y, x]

cv2.namedWindow("Player Tracking")
cv2.setMouseCallback("Player Tracking", on_click)

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        state["frame"] = frame

        results = model.track(frame, persist=True, classes=[0])
        state["boxes"] = results[0].boxes

    boxes = state["boxes"]
    annotated = state["frame"].copy()

    if boxes is not None and boxes.id is not None:
        hsv_frame = cv2.cvtColor(state["frame"], cv2.COLOR_BGR2HSV)

        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            x1, y1, x2, y2 = map(int, box)

            if target_color is None:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(annotated, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                roi = hsv_frame[max(0, y1):((y1 + y2) // 2), max(0, x1):x2]
                if roi.size == 0:
                    continue
                h, s, v = target_color
                lower = np.array([max(0, h - tolerance), 50, 50])
                upper = np.array([min(179, h + tolerance), 255, 255])
                mask = cv2.inRange(roi, lower, upper)
                if np.count_nonzero(mask) / mask.size > 0.1:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if paused:
        cv2.putText(annotated, "PAUSED — click a jersey, then press Space to resume", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    elif target_color is None:
        cv2.putText(annotated, "Press Space to pause and click a jersey color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(annotated, "Press R to reset color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Player Tracking", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        target_color = None
    elif key == ord(" "):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
