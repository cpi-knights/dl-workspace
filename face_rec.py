# import logging

# logging.basicConfig(level=logging.CRITICAL)

# import numpy as np

# np.seterr(all="ignore")  # Suppress NumPy warnings

import cv2
from deepface import DeepFace
import os

# Directory where the folders of each person with images are stored
persons_folder = "datasets/persons"


# Function to draw bounding boxes and labels
def draw_face(frame, box, name, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, name, (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Start webcam feed
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Convert to grayscale for OpenCV's face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's Haar Cascade for face detection (you can also use DNN for more accuracy)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for x, y, w, h in faces:
        # Extract the face from the frame
        face_image = frame[y: y + h, x: x + w]

        # Use DeepFace for face recognition
        try:
            # Use DeepFace to find matching faces
            result = DeepFace.find(
                face_image,
                db_path=persons_folder,
                enforce_detection=False,
                silent=True,
            )

            if result:
                # Get the path of the matched image (from the 'identity' field)
                matched_path = list(
                    set(
                        result[0]["identity"]
                        .map(lambda x: os.path.basename(os.path.dirname(x)))
                        .to_list()
                    )
                )

                # Extract the person's name from the folder name (the first part of the path)
                print("Name: ", matched_path)
                name = matched_path[0]  # Folder name is the person's name
                color = (0, 255, 0)  # Green for known
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
        except Exception as e:
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown

        # Draw the bounding box and name
        draw_face(frame, (x, y, w, h), name, color)

    # Show the frame
    cv2.imshow("Face Recognition Security", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
