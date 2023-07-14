#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:26:19 2023

@author: vivek
"""



import json
import cv2

import os
import sys
import numpy as np
import pandas as pd




DATASET_PATH =  "/home/vivek/VisionWorkspace/dataset/ICC WT20"
STROKES_PATH = "/home/vivek/VisionWorkspace/dataset/highlights_dataset_labels/sample_labels_shots/ICC WT20/"


def read_json(label_path):
    file = open(label_path, 'r')
    json_data = json.load(file)
    # print((json_data))
    return json_data
    
abc = read_json(STROKES_PATH+'ICC WT20 Ireland v Oman Match Highlights.json')
abc1 = list(abc.keys())[0]
abc2 = abc[abc1]

start = []
end = []

for i in abc2:
    # print(type(i))
    
    start.append(i[0])
    end.append(i[1])
            
print(start)
print(end)


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('/home/vivek/VisionWorkspace/dataset/ICC WT20/ICC WT20 Ireland v Oman Match Highlights.avi')

# Get total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 

# Get frames per second (fps) and frame size of the original video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Define starting and ending frame numbers for the clip
start_frame = 88  # Replace with the starting frame number for the clip
end_frame = 227  # Replace with the ending frame number for the clip

# Validate starting and end frame numbers
if start_frame < 0:
    start_frame = 0
if end_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT):
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the starting frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Create VideoWriter object to save the clip as a new video file
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # FourCC code for AVI format
output_path = "/home/vivek/VisionWorkspace/test/output_clip/ICC WT20 Ireland v Oman Match Highlights-Clip1.avi"  # Replace with the desired output path
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size, isColor=True)

# Loop through the frames and write them to the output file
for frame_number in range(start_frame, end_frame):
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if end of video is reached
        
    cv2.imshow("Video", frame)
    cv2.waitKey(25)  # Delay for 30 milliseconds


    # Write the frame to the output file
    out.write(frame)

 
# When everything done, release the video capture object
cap.release()

# Release the video file and the output file
out.release()
 
# Closes all the frames
cv2.destroyAllWindows()

# def read_video(video_path):
    
#     video = open(video_path, 'r')
#     video_data  = cv2.VideoCapture(())
    

# for cliping the video

# Validate starting and end frame numbers
# if start_frame < 0:
#     start_frame = 0
# if end_frame > total_frames:
#     end_frame = total_frames

# # Set the starting frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)



# # Check if camera opened successfully
# # if (cap.isOpened()== False): 
# #   print("Error opening video stream or file")
#  # Loop through the frames and play the video
# while True:
#     ret, frame = cap.read()

#     if not ret: # Check if camera opened successfully
# # if (cap.isOpened()== False): 
#
#         break  # Break the loop if end of video is reached

#     cv2.imshow("Video", frame)
#     cv2.waitKey(30)  # Delay for 30 milliseconds

#     # Stop playing the video when end frame is reached
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
#         break
    
    
# # Read until video is completed
# while(cap.isOpened()):
#   # Capture frame-by-frame
#   ret, frame = cap.read()
#   if ret == True:
 
#     # Display the resulting frame
#     cv2.imshow('Frame',frame)
 
#     # Press Q on keyboard to  exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
 
#   # Break the loop
#   else: 
#     break
