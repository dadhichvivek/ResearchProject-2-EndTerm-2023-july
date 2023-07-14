import json
import cv2
import os
import sys
import numpy as np
import pandas as pd

DATASET_PATH = "/home/vivek/VisionWorkspace/dataset/ICC WT20"
STROKES_PATH = "/home/vivek/VisionWorkspace/dataset/highlights_dataset_labels/sample_labels_shots/ICC WT20/"
OUTPUT_PATH = "/home/vivek/VisionWorkspace/vivek_clusters/test/output_clip/"

def read_json(label_path):
    file = open(label_path, 'r')
    json_data = json.load(file)
    return json_data

# Read the sorted video files in the dataset path
video_files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith('.avi')])

# Read the sorted JSON files in the strokes path/home/vivek/VisionWorkspace/vivek_clusters/test
json_files = sorted([f for f in os.listdir(STROKES_PATH) if f.endswith('.json')])

# Loop through each video file and its corresponding JSON file
for video_file, json_file in zip(video_files, json_files):
    
    # Get the video file properties
    video_path = os.path.join(DATASET_PATH, video_file)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Read the JSON file containing the start and end times of video clips
    json_path = os.path.join(STROKES_PATH, json_file)
    clip_labels = read_json(json_path)
    clip_key = list(clip_labels.keys())[0]
    clip_data = clip_labels[clip_key]

    start = []
    end = []

    # Extract the start and end times for each clip from the JSON file
    for clip in clip_data:
        start.append(clip[0])
        end.append(clip[1])

    # Loop through each clip in the JSON file and extract it from the video file
    for clip_index in range(len(start)):
        # Get the start and end frames for the selected clip
        start_frame = int(start[clip_index] * fps)
        end_frame = int(end[clip_index] * fps)

        # Validate starting and end frame numbers
        if start_frame < 0:
            start_frame = 0
        if end_frame > total_frames:
            end_frame = total_frames

        # Set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create VideoWriter object to save the clip as a new video file
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.join(DATASET_PATH, clip_key + '-Clip' + str(clip_index+1) + '.avi')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size, isColor=True)

        # Loop through the frames and write them to the output file
        for frame_number in range(start_frame, end_frame):
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Video", frame)
            cv2.waitKey(40)

            # Write the frame to the output file
            out.write(frame)

        # Release the output writer object
        out.release()

    # Release the video capture object
    cap.release()

# Close all the frames
cv2.destroyAllWindows()
