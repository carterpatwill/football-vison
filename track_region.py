import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("vids/FootballVid.mp4")

region = {"start": None, "end": None, "drawing": False, "set": False}
paused = False
state = {"frame": None, "boxes": None}

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        region["start"] = (x, y)
        region["drawing"] = True
        region["set"] = False
    elif event == cv2.EVENT_MOUSEMOVE and region["drawing"]:
        region["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        region["end"] = (x, y)
        region["drawing"] = False
        region["set"] = True

cv2.namedWindow("Player Tracking")
cv2.setMouseCallback("Player Tracking", on_mouse)

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
        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            in_region = True
            if region["set"] and region["start"] and region["end"]:
                rx1 = min(region["start"][0], region["end"][0])
                ry1 = min(region["start"][1], region["end"][1])
                rx2 = max(region["start"][0], region["end"][0])
                ry2 = max(region["start"][1], region["end"][1])
                in_region = rx1 <= cx <= rx2 and ry1 <= cy <= ry2

            if in_region:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if region["start"] and region["end"]:
        cv2.rectangle(annotated, region["start"], region["end"], (255, 0, 0), 2)

    if paused:
        cv2.putText(annotated, "PAUSED — click and drag to draw a region, then press Space", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    elif not region["set"]:
        cv2.putText(annotated, "Press Space to pause and draw a region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(annotated, "Press R to reset region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Player Tracking", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        region["start"] = None
        region["end"] = None
        region["set"] = False
    elif key == ord(" "):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
