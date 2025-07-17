import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
JOINTS = [11,12,13,14,15,16,19,20,23,24,25,26,27,28]
POSE_CONNECTIONS = [
    #torsp
    (11, 12),  # left_shoulder to right_shoulder
    (11, 23),  # left_shoulder to left_hip
    (12, 24),  # right_shoulder to right_hip
    (23, 24),  # left_hip to right_hip

    # Left Arm
    (11, 13),  # L_brachium
    (13, 15),  # L_antebrachium
    (15, 19),  # left_wrist to left_index

    # Right Arm
    (12, 14),  # right_shoulder to right_elbow
    (14, 16),  # right_elbow to right_wrist
    (16, 20),  # right_wrist to right_index

    # Left Leg
    (23, 25),  # left_hip to left_knee
    (25, 27),  # left_knee to left_ankle

    # Right Leg
    (24, 26),  # right_hip to right_knee
    (26, 28),  # right_knee to right_ankle

]
def get_coordinates(landmark,shape):
    h,w = shape
    x=int(landmark.x*w)
    y=int(landmark.y*h)
    return (x,y)
def draw_parts(frame, pt1 : tuple,pt2 : tuple, color=(0,255,0)):
    cv2.line(frame,pt1,pt2,color, thickness=20)
    


while True:
    ret, frame = cap.read()
    if not ret: 
        break
    frame = cv2.flip(frame,flipCode=1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        #body_parts
        for str_limb,end_limb in POSE_CONNECTIONS:
            str=get_coordinates(landmarks[str_limb],(h,w))
            end=get_coordinates(landmarks[end_limb],(h,w))
            draw_parts(black_canvas,str,end)
        #face
        nose=landmarks[0]
        cx =int(nose.x*w)
        cy = int(nose.y*h)
        cv2.circle(black_canvas,(cx,cy),40,(0,255,0),thickness=-1)

        #joints
        for pts in JOINTS:
            pt = get_coordinates(landmarks[pts],(h,w))
            cv2.circle(black_canvas,pt,15,(0,0,0),thickness = -1)

    cv2.imshow("sim",black_canvas)
    if cv2.waitKey(1) and 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
        