import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import os

os.makedirs("output", exist_ok=True)

TEAM_COLORS = {
    1: (0, 0, 255),    # red (BGR)
    2: (30, 30, 30),   # black
}

def get_color(tid):
    return TEAM_COLORS.get(team_assignments.get(tid), (180, 180, 180))

def save_trails(frame_shape):
    canvas = np.full(frame_shape, 255, dtype='uint8')
    canvas[field_edges > 0] = (200, 200, 200)  # light grey field edges
    for tid in selected_ids:
        color = get_color(tid)
        pts = trails[tid]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 4)
    filename = f"output/trails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, canvas)
    print(f"Saved: {filename}")

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("vids/FootbalVid24k.mp4")

ret, first_frame = cap.read()
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
field_edges = cv2.Canny(gray, 50, 150)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

selected_ids = set()
team_assignments = {}  # tid -> 1 or 2
trails = defaultdict(list)
paused = False
active_team = 1  # which team clicking assigns to
state = {"frame": None, "boxes": None}

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and state["boxes"] is not None:
        boxes = state["boxes"]
        if boxes.id is not None:
            for box, tid in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    tid = int(tid)
                    if tid in selected_ids and team_assignments.get(tid) == active_team:
                        selected_ids.remove(tid)
                        team_assignments.pop(tid, None)
                        trails[tid].clear()
                    else:
                        selected_ids.add(tid)
                        team_assignments[tid] = active_team
                    break

cv2.namedWindow("Player Tracking")
cv2.setMouseCallback("Player Tracking", on_click)

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        state["frame"] = frame

        results = model.track(frame, persist=True, classes=[0], imgsz=1280, conf=0.15, tracker="custom_tracker.yaml")
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

    team_label = "RED team" if active_team == 1 else "BLACK team"
    indicator_color = (0, 0, 255) if active_team == 1 else (50, 50, 50)

    if paused:
        cv2.putText(annotated, f"PAUSED [{team_label}] — click to assign, 1/2 to switch team", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, indicator_color, 2)
    elif not selected_ids:
        cv2.putText(annotated, "Press Space to pause — press 1 or 2 to pick a team", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(annotated, f"[{team_label}] {len(selected_ids)} player(s) — Space to pause, R to reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, indicator_color, 2)

    cv2.imshow("Player Tracking", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        save_trails(state["frame"].shape)
        break
    elif key == ord("s"):
        save_trails(state["frame"].shape)
    elif key == ord("r"):
        selected_ids.clear()
        team_assignments.clear()
        trails.clear()
    elif key == ord(" "):
        paused = not paused
    elif key == ord("1"):
        active_team = 1
    elif key == ord("2"):
        active_team = 2

save_trails(state["frame"].shape)
cap.release()
cv2.destroyAllWindows()
