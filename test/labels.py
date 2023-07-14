#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:06:13 2023

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

def read_video(video_path):
    video = open(video_path, 'r')
    video_data  = cv2.VideoCapture(())
    
    
# if __name__ == '__main__':
    
#     videonames = sorted(os.listdir(DATASET_PATH))
#     labels = sorted(os.listdir(STROKES_PATH))
    
#     print(videonames)
#     print(labels)
     