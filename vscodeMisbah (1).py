# from ultralytics import YOLO
# import cv2

# model = YOLO('best.pt')

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("Cannot access webcam"); exit()

# cv2.namedWindow("YOLOv8 - Real Time Detection", cv2.WINDOW_NORMAL)

# while True:
#     ret, frame = cap.read()
#     if not ret: break

#     # YOLOv8 will show its own window if show=True
#     results = model.predict(source=frame, conf=0.3, show=True)

#     # Optional: also show with OpenCV
#     # annotated = results[0].plot()  
#     # cv2.imshow("YOLOv8 - Real Time Detection", annotated)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()













# from ultralytics import YOLO
# import cv2
# from playsound import playsound
# import threading

# # Load the trained YOLO model
# model = YOLO('best.pt')

# # Start webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("Cannot access webcam")
#     exit()

# cv2.namedWindow("YOLOv8 - Real Time Detection", cv2.WINDOW_NORMAL)

# # Function to play beep sound in a separate thread
# def play_beep():
#     threading.Thread(target=playsound, args=('beep.mp3',), daemon=True).start()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run detection
#     results = model(frame, conf=0.8)

#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             Yeh line green box draw karti hai cigarette ke aas-paas.
# #  aur 2 is thickness of the box line.


#             label = f"Cigarette: {conf:.2f}"
#             # Yeh ek label create karta hai, jaise: Cigarette: 0.89
#             cv2.putText(frame, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             # Play mp3 beep if cigarette (class 0) is detected
#             if cls_id == 0:
#                 play_beep()

   

#     # Quit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
from playsound import playsound
import threading

# Load the trained YOLO model
model = YOLO('best.pt')

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot access webcam")
    exit()

cv2.namedWindow("YOLOv8 - Real Time Detection", cv2.WINDOW_NORMAL)

# Function to play beep sound in a separate thread
def play_beep():
    threading.Thread(target=playsound, args=('beep.mp3',), daemon=True).start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=0.6)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Yeh line green box draw karti hai cigarette ke aas-paas.
                # aur 2 is thickness of the box line.

                label = f"Cigarette: {conf:.2f}"
                # Yeh ek label create karta hai, jaise: Cigarette: 0.89
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Play mp3 beep if cigarette (class 0) is detected
                if cls_id == 0:
                    play_beep()

        # Show annotated frame
        cv2.imshow("YOLOv8 - Real Time Detection", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Manually interrupted. Exiting...")

finally:
    # Cleanup (always runs, even on error or Ctrl+C)
    cap.release()
    cv2.destroyAllWindows()
