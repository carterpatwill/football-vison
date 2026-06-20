import cv2
import numpy as np
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Install with: pip install sahi
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8m.pt",
    confidence_threshold=0.25,
    device="cpu",
)

cap = cv2.VideoCapture("vids/FootballVid.mp4")

selected_ids = set()
trails = defaultdict(list)
paused = False
state = {"frame": None, "track_results": {}}

COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 165, 0), (255, 255, 0), (128, 0, 255),
]

def get_color(tid):
    return COLORS[tid % len(COLORS)]


class SimpleTracker:
    def __init__(self, max_dist=80, max_lost=30):
        self.next_id = 0
        self.tracks = {}
        self.lost = {}
        self.max_dist = max_dist
        self.max_lost = max_lost

    def update(self, detections):
        centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in detections]

        for tid in list(self.lost):
            self.lost[tid] += 1
            if self.lost[tid] > self.max_lost:
                del self.tracks[tid]
                del self.lost[tid]

        if not detections:
            return {}

        if not self.tracks:
            for i, c in enumerate(centroids):
                self.tracks[self.next_id] = (c, detections[i])
                self.lost[self.next_id] = 0
                self.next_id += 1
            return {tid: data[1] for tid, data in self.tracks.items()}

        track_ids = list(self.tracks.keys())
        track_cents = [self.tracks[t][0] for t in track_ids]

        dists = []
        for i, tc in enumerate(track_cents):
            for j, dc in enumerate(centroids):
                d = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5
                dists.append((d, i, j))
        dists.sort()

        used_tracks = set()
        used_dets = set()
        result = {}

        for d, ti, di in dists:
            if d > self.max_dist:
                break
            if ti in used_tracks or di in used_dets:
                continue
            tid = track_ids[ti]
            self.tracks[tid] = (centroids[di], detections[di])
            self.lost[tid] = 0
            result[tid] = detections[di]
            used_tracks.add(ti)
            used_dets.add(di)

        for di, det in enumerate(detections):
            if di not in used_dets:
                self.tracks[self.next_id] = (centroids[di], det)
                self.lost[self.next_id] = 0
                result[self.next_id] = det
                self.next_id += 1

        return result


tracker = SimpleTracker()


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and state["track_results"]:
        for tid, (x1, y1, x2, y2) in state["track_results"].items():
            if x1 <= x <= x2 and y1 <= y <= y2:
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

        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        detections = [
            (int(o.bbox.minx), int(o.bbox.miny), int(o.bbox.maxx), int(o.bbox.maxy))
            for o in result.object_prediction_list
            if o.category.id == 0
        ]

        track_results = tracker.update(detections)
        state["track_results"] = track_results

        for tid, (x1, y1, x2, y2) in track_results.items():
            if tid in selected_ids:
                trails[tid].append(((x1 + x2) // 2, (y1 + y2) // 2))

    annotated = state["frame"].copy()
    track_results = state["track_results"]

    for tid in selected_ids:
        color = get_color(tid)
        pts = trails[tid]
        for i in range(1, len(pts)):
            cv2.line(annotated, pts[i - 1], pts[i], color, 4)

    for tid, (x1, y1, x2, y2) in track_results.items():
        if not selected_ids or tid in selected_ids:
            color = get_color(tid) if tid in selected_ids else (180, 180, 180)
            thickness = 2 if tid in selected_ids else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if paused:
        cv2.putText(annotated, "PAUSED — click players to select/deselect, Space to resume", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif not selected_ids:
        cv2.putText(annotated, "[SAHI] Press Space to pause and select players", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
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
