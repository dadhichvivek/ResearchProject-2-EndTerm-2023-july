#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:44:38 2023

@author: vivek
"""


import cv2
import numpy as np


# Function to calculate HOOF for a given frame
def calculate_hoof(frame_prev, frame_next, bins=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False):
    # Convert frames to grayscale
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Convert optical flow from Cartesian to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Split the angle into multiple bins
    bin_width = 2 * np.pi / bins
    angle_bins = (angle / bin_width).astype(int)
    
    # Initialize HOOF histogram
    histogram = np.zeros(bins)
    
    # Loop through each cell
    for i in range(0, gray_prev.shape[0], pixels_per_cell[1]):
        for j in range(0, gray_prev.shape[1], pixels_per_cell[0]):
            # Get the magnitude and angle of the optical flow in the current cell
            magnitudes_cell = magnitude[i:i + pixels_per_cell[1], j:j + pixels_per_cell[0]]
            angles_cell = angle_bins[i:i + pixels_per_cell[1], j:j + pixels_per_cell[0]]
            
            # Flatten the magnitudes and angles
            magnitudes_cell = magnitudes_cell.flatten()
            angles_cell = angles_cell.flatten()
            
            # Check for zero values in magnitudes_cell
            if np.any(magnitudes_cell == 0):
                continue
            
            # Compute the histogram for the current cell
            histogram_cell = np.histogram(angles_cell, bins=bins, range=(0, bins))[0]
            
            # Add the histogram of the current cell to the overall HOOF histogram
            histogram += histogram_cell * magnitudes_cell.mean()
    
    # Normalize the histogram
    histogram /= np.sum(histogram)
    
    # Visualize the histogram if requested
    if visualize:
        import matplotlib.pyplot as plt
        plt.bar(np.arange(bins), histogram)
        plt.xlabel('Angle Bins')
        plt.ylabel('Normalized Magnitude')
        plt.show()
    
    return histogram

# Open video file
cap = cv2.VideoCapture('/home/vivek/VisionWorkspace/test/output_clip/ICC WT20 Ireland v Oman Match Highlights-Clip1.avi')

# Read the first frame
ret, frame_prev = cap.read()

# Convert the first frame to grayscale
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

# Initialize the HOOF histogram for the first frame
histogram_prev = calculate_hoof(frame_prev, frame_prev)

while True:
    # Read the next frame
    ret, frame_next = cap.read()
    
    # Break the loop if video has ended
    if not ret:
        break
    
    # Calculate HOOF for the current frame
    histogram_next = calculate_hoof(frame_prev, frame_next)
    
    # Convert histograms to the same data type and depth
    histogram_prev = histogram_prev
    
    histogram_prev = histogram_prev.astype(np.float32)
    histogram_next = histogram_next.astype(np.float32)

    # Calculate the histogram distance (e.g., Chi-Squared distance) between the histograms of the current and previous frames
    histogram_distance = cv2.compareHist(histogram_prev, histogram_next, cv2.HISTCMP_CHISQR)

    # Print the histogram distance
    print("Histogram Distance: ", histogram_distance)

    # Update the previous frame histogram
    histogram_prev = histogram_next

    # Show the current frame
    cv2.imshow('Frame', frame_next)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


