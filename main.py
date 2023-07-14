#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:21:30 2023

@author: vivek
"""


import cv2
import os
import numpy as np
import pandas as pd
import sys
import extract_hoof_feats as hoof

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


if __name__ == '__main__':
    
    videonames = sorted(os.listdir(DATASET_PATH))
    labels = sorted(os.listdir(STROKES_PATH))
    
    
    # cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    
    
    
    
    
    
    



