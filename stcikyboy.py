import cv2
import mediapipe as mp
import numpy as np
# Init mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Custom drawing style: green lines, thick joints
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=6)

# Open webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert image
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Prepare black background
        black_canvas = np.zeros_like(frame)

        # Draw pose on black canvas
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                black_canvas,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        cv2.imshow('Green Stickman Pose Mirror', black_canvas)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
