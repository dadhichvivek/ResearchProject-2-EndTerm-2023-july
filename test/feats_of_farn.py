#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:28:40 2023

@author: vivek
"""

import cv2

# Load input video
cap = cv2.VideoCapture('/home/vivek/VisionWorkspace/test/output_clip/ICC WT20 Ireland v Oman Match Highlights-Clip1.avi')

import numpy as np

# Parameters for Farneback optical flow
pyr_scale = 0.5 # pyramid scale factor
levels = 3 # number of pyramid levels
winsize = 15 # window size for optical flow calculation
iterations = 3 # number of iterations for optical flow calculation
poly_n = 5 # size of the pixel neighborhood used to find polynomial expansion in each pixel
poly_sigma = 1.1 # standard deviation of the Gaussian that is used to smooth derivatives
flags = 0 # optional flags, usually set to 0

# Initialize previous grayscale frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Capture frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev=prev_gray, next=gray, flow=None, pyr_scale=pyr_scale,
                                        levels=levels, winsize=winsize, iterations=iterations,
                                        poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)

    # Visualize optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    optical_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display frames
    cv2.imshow('Original', frame)
    cv2.imshow('Optical Flow', optical_flow)

    # Update previous frame
    prev_gray = gray

    # Exit loop on 'q' key press
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
