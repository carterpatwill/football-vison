import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("vids/FootballVid.mp4")

selected_ids = set()
trails = defaultdict(list)
paused = False
state = {"frame": None, "boxes": None}

COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 165, 0), (255, 255, 0), (128, 0, 255),
]

def get_color(tid):
    return COLORS[tid % len(COLORS)]

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and state["boxes"] is not None:
        boxes = state["boxes"]
        if boxes.id is not None:
            for box, tid in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    tid = int(tid)
                    if tid in selected_ids:
                        selected_ids.remove(tid)
                        trails[tid].clear()
                    else:
                        selected_ids.add(tid)
                    break

cv2.namedWindow("Player Tracking")
cv2.setMouseCallback("Player Tracking", on_click)

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        state["frame"] = frame

        results = model.track(frame, persist=True, classes=[0], imgsz=1280)
        state["boxes"] = results[0].boxes

        boxes = state["boxes"]
        if boxes is not None and boxes.id is not None:
            for box, tid in zip(boxes.xyxy, boxes.id):
                tid = int(tid)
                if tid in selected_ids:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trails[tid].append((cx, cy))

    boxes = state["boxes"]
    annotated = state["frame"].copy()

    for tid in selected_ids:
        color = get_color(tid)
        pts = trails[tid]
        for i in range(1, len(pts)):
            cv2.line(annotated, pts[i - 1], pts[i], color, 4)

    if boxes is not None and boxes.id is not None:
        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            x1, y1, x2, y2 = map(int, box)
            if not selected_ids or tid in selected_ids:
                color = get_color(tid) if tid in selected_ids else (180, 180, 180)
                thickness = 2 if tid in selected_ids else 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if paused:
        cv2.putText(annotated, "PAUSED — click players to select/deselect, Space to resume", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif not selected_ids:
        cv2.putText(annotated, "[HIGH RES] Press Space to pause and select players", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(annotated, f"Tracking {len(selected_ids)} player(s) — Space to pause, R to reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Player Tracking", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        selected_ids.clear()
        trails.clear()
    elif key == ord(" "):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
