#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:40:13 2023

@author: vivek
"""

import cv2
import os
import sys


DATASET_PATH = "/home/vivek/Downloads/ICC WT20"

if __name__ == "__main__":
    
    files = os.listdir(DATASET_PATH)
    
    for file in files:
        
        filepath = os.path.join(DATASET_PATH, file)
        
        print(filepath)