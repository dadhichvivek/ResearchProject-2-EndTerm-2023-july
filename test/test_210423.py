#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:37:45 2023

@author: vivek
"""

import cv2
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Load the video
video_path = '/home/vivek/VisionWorkspace/test/output_clip/ICC WT20 Ireland v Oman Match Highlights-Clip1.avi'
cap = cv2.VideoCapture(video_path)

# Define the number of frames to sample and the number of clusters
n_frames = 50
n_clusters = 5

# Extract color histograms from sampled frames
frame_indices = np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, n_frames, dtype=int)
frames = []
for i in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    else:
        break
frames = np.array(frames)
features = np.zeros((n_frames, 4096))
for i, frame in enumerate(frames):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    features[i] = hist.flatten()

# Perform spectral clustering
clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10)
labels = clustering.fit_predict(features)

# Visualize the clustering results
fig, axs = plt.subplots(1, n_clusters, figsize=(30, 10))
for i in range(n_clusters):
    cluster_frames = frames[labels == i]
    axs[i].imshow(np.vstack(cluster_frames))
    axs[i].axis('off')
plt.show()