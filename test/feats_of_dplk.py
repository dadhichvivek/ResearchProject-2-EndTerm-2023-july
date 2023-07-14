 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:30:41 2023

@author: vivek
"""

import json
import cv2

import os
import sys
import numpy as np
import pandas as pd

video_path = '/home/vivek/VisionWorkspace/test/output_clip/ICC WT20 Ireland v Oman Match Highlights-Clip1.avi'

cap = cv2.VideoCapture(video_path)


# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

while True:
    # Capture two consecutive frames from the video
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if not ret:
        break

    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Dense Pyramid Lucas-Kanade
    p1 = cv2.goodFeaturesToTrack(frame1_gray, 500, 0.01, 10)
    p2, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, p1, None, **lk_params)

    # Select good points
    good_new = p2[st == 1]
    good_old = p1[st == 1]

    # Draw optical flow lines on the first frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        center = (int(a), int(b))
        frame1 = cv2.circle(frame1, center, 5, (0, 255, 0), -1)
        frame1 = cv2.line(frame1, center, (int(c), int(d)), (0, 0, 255), 2)

    # Display the first frame with optical flow lines
    cv2.imshow('Dense Pyramid Lucas-Kanade Optical Flow', frame1)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

