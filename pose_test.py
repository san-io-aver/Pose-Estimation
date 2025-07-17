import cv2
import mediapipe as mp
import numpy as np
import math
cap = cv2.VideoCapture(0)
mp_pose  = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

reps=0
direction_is_up = False

def getAngle(A,B,C)-> float:
    a=np.array(A)
    b=np.array(B)
    c=np.array(C)

    angle_rad = np.arctan2(a[1]-b[1],a[0]-b[0]) - np.arctan2(c[1]-b[1],c[0]-b[0])
    angle = np.abs(angle_rad * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*w,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*h]
                          
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x*w,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y*h]
        
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x*w,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y*h]

        
        angle = getAngle(right_shoulder,right_elbow,right_wrist)

        if angle>160 and direction_is_up:
            reps += 1
            direction_is_up = False
        if angle<45 :
            direction_is_up = True


        cv2.putText(frame,f"Angle: {angle}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.putText(frame,f"Reps: {reps}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),3)
         
    cv2.imshow('Workout Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
