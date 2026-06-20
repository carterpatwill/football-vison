import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("vids/FootballVid.mp4")

selected_id = None
paused = False
state = {"boxes": None, "frame": None}

def on_click(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN and state["boxes"] is not None:
        boxes = state["boxes"]
        if boxes.id is not None:
            for box, tid in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_id = int(tid)
                    break

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
        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            x1, y1, x2, y2 = map(int, box)
            if selected_id is None or tid == selected_id:
                color = (0, 255, 0) if tid == selected_id else (0, 200, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if paused:
        cv2.putText(annotated, "PAUSED — click a player, then press Space to resume", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    elif selected_id is None:
        cv2.putText(annotated, "Press Space to pause and select a player", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(annotated, f"Tracking ID {selected_id} — press R to reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Player Tracking", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        selected_id = None
    elif key == ord(" "):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
