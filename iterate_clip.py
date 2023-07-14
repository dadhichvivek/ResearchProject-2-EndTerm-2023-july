#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:04:53 2023

@author: vivek
"""

import cv2

from iterate_strokes import get_clips

DATASET_PATH = "/home/vivek/VisionWorkspace/dataset/ICC WT20"
STROKES_PATH = "/home/vivek/VisionWorkspace/dataset/highlights_dataset_labels/sample_labels_shots/ICC WT20/"

# Generate the clips list
clips = get_clips(DATASET_PATH, STROKES_PATH)

# Get the first clip from the clips list
filename, clip_key, clip_index, clip_frames = clips[0]

# Loop through each frame in the clip and display it
for frame in clip_frames:
    cv2.imshow('Clip', frame)
    cv2.waitKey(25) # Delay in milliseconds
    if cv2.getWindowProperty('Clip', cv2.WND_PROP_VISIBLE) < 1:
        break

# Close the window
cv2.destroyAllWindows()
