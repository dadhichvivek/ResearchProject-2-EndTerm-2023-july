import cv2
import json
import os
from tqdm import tqdm

def read_json(label_path):
    file = open(label_path, 'r')
    json_data = json.load(file)
    return json_data

def get_clips(data_path, stroke_path):
    video_files = sorted([f for f in os.listdir(data_path) if f.endswith('.avi')])
    json_files = sorted([f for f in os.listdir(stroke_path) if f.endswith('.json')])
    clips = []
    total_clips = 0
    stroke_features = []

    for video_file, json_file in tqdm(zip(video_files, json_files), total=len(video_files)):
        video_path = os.path.join(data_path, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        json_path = os.path.join(stroke_path, json_file)
        clip_labels = read_json(json_path)
        clip_key = list(clip_labels.keys())[0]
        clip_data = clip_labels[clip_key]
        start = [clip[0] for clip in clip_data]
        end = [clip[1] for clip in clip_data]
        orb = cv2.ORB_create(nfeatures=1500)

        for clip_index in range(len(start)):
            start_frame = int(start[clip_index])
            end_frame = int(end[clip_index])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            clip = []

            for frame_number in range(start_frame, end_frame):
                ret, frame = cap.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints_orb, descriptors = orb.detectAndCompute(frame_gray, None)
                desc_2vec = descriptors[0:2, :].flatten()

                if not ret:
                    break

                clip.append(desc_2vec)

            total_clips += 1
            stroke_features.append(clip)

        cap.release()

    return stroke_features

def extract_and_display_stroke_features(data_path, stroke_path):
    # Generate the clips list
    clips = get_clips(data_path, stroke_path)

    # Get the first clip from the clips list
    filename, clip_key, clip_index, clip_frames = clips[0]

    # Loop through each frame in the clip and display it
    for frame in clip_frames:
        cv2.imshow('Clip', frame)
        cv2.waitKey(25)  # Delay in milliseconds
        if cv2.getWindowProperty('Clip', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close the window
    cv2.destroyAllWindows()

# Example usage
STROKES_PATH = "E:/Research Project/Clustring Cricket Video/Data/highlights_dataset_labels/sample_labels_shots/ICC WT20"
DATASET_PATH = "E:/Research Project/Clustring Cricket Video/Data/ICC WT20.tar/ICC WT20/"

extract_and_display_stroke_features(DATASET_PATH, STROKES_PATH)
