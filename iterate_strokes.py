import json
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# DATASET_PATH = "/home/vivek/VisionWorkspace/dataset/ICC WT20"
# STROKES_PATH = "/home/vivek/VisionWorkspace/dataset/highlights_dataset_labels/sample_labels_shots/ICC WT20/"

def read_json(label_path):
    file = open(label_path, 'r')
    json_data = json.load(file)
    return json_data

def get_clips(data_path, stroke_path):
    # Read the sorted video files in the dataset path
    video_files = sorted([f for f in os.listdir(data_path) if f.endswith('.avi')])

    # Read the sorted JSON files in the strokes path
    json_files = sorted([f for f in os.listdir(stroke_path) if f.endswith('.json')])

    clips = []

    # Counter to keep track of total number of clips
    total_clips = 0
    stroke_features = []

    # Loop through each video file and its corresponding JSON file
    for video_file, json_file in tqdm(zip(video_files, json_files), total=len(video_files)):

        # Get the video file properties
        video_path = os.path.join(data_path, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Read the JSON file containing the start and end times of video clips
        json_path = os.path.join(stroke_path, json_file)
        clip_labels = read_json(json_path)
        clip_key = list(clip_labels.keys())[0]
        clip_data = clip_labels[clip_key]

        start = []
        end = []

        # # Extract the start and end times for each clip from the JSON file
        # for clip in clip_data:
        #     start.append(clip[0])
        #     end.append(clip[1])

        start = [clip[0] for clip in clip_data]
        end = [clip[1] for clip in clip_data]
        
        
        orb = cv2.ORB_create(nfeatures=1500)

        # Loop through each clip in the JSON file and extract it from the video file
        for clip_index in range(len(start)):
            # Get the start and end frames for the selected clip
            start_frame = int(start[clip_index])
            end_frame = int(end[clip_index])

            # # Validate starting and end frame numbers
            # if start_frame < 0:
            #     start_frame = 0
            # if end_frame > total_frames:
            #     end_frame = total_frames

            # Set the starting frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            clip = []

            # Loop through the frames and add them to the clip
            for frame_number in range(start_frame, end_frame):
                ret, frame = cap.read()
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # feature extraction
                # vec = cv2.extract_feature(frame)
                keypoints_orb, descriptors = orb.detectAndCompute(frame_gray, None)

                desc_2vec = descriptors[0:2,:].flatten()
                
                if not ret:
                    break

                # Add the frame to the clip
                clip.append(desc_2vec)

            # Add the clip to the list with the filename and index of start and end list
            #clips.append((video_file, clip_key, clip_index, clip))

            # Increment the total clips counter
            total_clips += 1
            stroke_features.append(clip)

        # Release the video capture object
        cap.release()
        break

    # Return the clips list
    return stroke_features

# Call the get_clips function to generate the clips list
# clips = get_clips(DATASET_PATH, STROKES_PATH)

# Print the total number of clips
# print(f"Total number of clips: {len(clips)}")

# Close all the frames
cv2.destroyAllWindows()


if __name__ == "__main__":
    # clips = get_clips()

    # Display the video names with the corresponding clip index and filename
    # for filename, clip_key, clip_index, clip in clips:
    #     print(f"{filename} - {clip_key} - {clip_index+1}")

    # Close all the frames
    cv2.destroyAllWindows()
