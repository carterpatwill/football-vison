import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # downloads automatically on first run

cap = cv2.VideoCapture("vids/FootballVid.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    results = model.track(frame, persist=True, classes=[0])  # class 0 = person

    annotated = results[0].plot()  # draws boxes + track IDs on the frame

    cv2.imshow("Player Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
