#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:10:57 2023

@author: vivek
"""

import cv2
import os
import sys


DATASET_PATH = "/home/vivek/VisionWorkspace/dataset/ICC WT20"

if __name__ == "__main__":
    
    files = os.listdir(DATASET_PATH)
    
    for file in files:
        count = 0
        filepath = os.path.join(DATASET_PATH, file)
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            ret, frame = cap.read()
        while ret and cap.isOpened():
            
            
            ret, frame = cap.read()
            count +=1
            
            if ret:
                cv2.imshow("Frames", frame)
                #print("{} : {}".format(count, frame.shape))
            if cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        break
            


