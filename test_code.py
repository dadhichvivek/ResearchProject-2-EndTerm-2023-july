#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 #   """
   # Created on Mon Apr 10 20:44:38 2023
    
   # @author: vivek
   # """

#import torch

import cv2
import os
import numpy as np
import pandas as pd
import sys

#import extract_hoof_feats as hoof
import json

import glob
import pickle



DATASET_PATH =  "/home/vivek/VisionWorkspace/dataset/ICC WT20"
STROKES_PATH = "/home/vivek/VisionWorkspace/dataset/highlights_dataset_labels/sample_labels_shots/ICC WT20"
CLUSTER_LABELS = ""


def read_labels(labels_path):
    '''
    Use complete path of file to extract the file contents (it will be a json file). 

    Parameters
    ----------
    labels_path : str
        Complete file to a json file.

    Returns
    -------
    list of tuples

    '''

def read_stroke_labels(self, videonames):
    
    vid_strokes  = []
    for video in video_list:
        vidname = video.rsplit('/',1)[1]
        stroke_file = os.path.join(self.stroke_dir, vidname.rsplit('.',1)[0]+'.json')
        assert os.path.isfile(stroke_file), "File not found {}".format(stroke_file)
        with open(stroke_file, 'r') as fp:
            strokes = json.load(fp)
        vid_strokes.append(strokes[list(strokes.key())[0]])
        print(vid_strokes)
    return vid_strokes

if __name__ == '__main__':
    
    videonames = sorted(os.listdir(DATASET_PATH))
    labels = sorted(os.listdir(STROKES_PATH))
    
    
    # cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                                           
            

                    
       