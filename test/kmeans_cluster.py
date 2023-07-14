#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:38:30 2023

@author: vivek
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean

# Load input video
cap = cv2.VideoCapture('/home/vivek/VisionWorkspace/dataset/ICC WT20/ICC WT20 - Afghanistan vs South Africa - Match Highlights.avi')

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

# Initialize empty list to store optical flow vectors
optical_flow_vectors = []

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

    # Flatten optical flow vectors
    optical_flow_vectors.append(flow.reshape(-1))

    # Update previous frame
    prev_gray = gray

    # Display frames
    cv2.imshow('Original', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Convert list of optical flow vectors to numpy array
optical_flow_vectors = np.array(optical_flow_vectors)

# Apply k-means clustering to optical flow vectors
kmeans = KMeans(n_clusters=5, random_state=0).fit(optical_flow_vectors)
kmeans_labels = kmeans.labels_

# Apply spectral clustering to optical flow vectors
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=10).fit(optical_flow_vectors)
spectral_labels = spectral.labels_

# Plot clustering result
import matplotlib.pyplot as plt

# Define color map for clustering labels
cmap = plt.cm.get_cmap('viridis', 3)

# Plot k-means clustering result
plt.figure(figsize=(10,5))
plt.scatter(optical_flow_vectors[:, 0], optical_flow_vectors[:, 1], c=kmeans_labels, cmap=cmap)
plt.title('K-means clustering')
plt.xlabel('Optical flow vector 1')
plt.ylabel('Optical flow vector 2')
plt.show()

# Plot spectral clustering result
plt.figure(figsize=(10,5))
plt.scatter(optical_flow_vectors[:, 0], optical_flow_vectors[:, 1], c=spectral_labels, cmap=cmap)
plt.title('Spectral clustering')
plt
