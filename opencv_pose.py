#!/usr/bin/env python

import cv2
import numpy as np

# MobileNet body keypoints
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Download model if needed
import os
if not os.path.exists('pose_iter_440000.caffemodel'):
    print("Downloading model...")
    os.system('wget -q http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_440000.caffemodel')
    os.system('wget -q https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_mpi.prototxt')

net = cv2.dnn.readNetFromCaffe('openpose_pose_mpi.prototxt', 'pose_iter_440000.caffemodel')

cap = cv2.VideoCapture(0)
inWidth = 368
inHeight = 368

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.1 else None)
    
    # Draw skeleton
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 255), 3)
            cv2.circle(frame, points[idFrom], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[idTo], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    cv2.imshow('OpenCV Pose', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
