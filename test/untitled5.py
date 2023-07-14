#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:28:40 2023

@author: vivek
"""

import cv2

# Load input video
cap = cv2.VideoCapture('/home/vivek/VisionWorkspace/test/output_clip/ICC WT20 Ireland v Oman Match Highlights-Clip1.avi')

# Define parameters for Farneback optical flow
pyr_scale = 0.5
levels = 5
winsize = 11
iterations = 5
poly_n = 5
poly_sigma = 1.1
flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Check if frame was read successfully
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If it's the first frame, initialize previous frame and points
    if 'prev_gray' not in locals():
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        continue

    # Calculate optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

    # Extract magnitude and angle of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Convert angle to hue value
    hue = angle * 180 / (2 * cv2.cv2.CV_PI)

    # Normalize magnitude to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert magnitude and hue to 8-bit unsigned integers
    magnitude = cv2.convertScaleAbs(magnitude)
    hue = cv2.convertScaleAbs(hue)

    # Merge magnitude and hue channels into an RGB image
    hsv = cv2.merge([hue, 255, magnitude])

    # Convert HSV image to BGR for display
    optical_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display optical flow
    cv2.imshow('Optical Flow', optical_flow)

    # Update previous frame and points
    prev_gray = gray
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
