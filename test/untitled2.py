#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:41:13 2023

@author: vivek
"""

import torch


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

import cv2
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

DATASET_PATH = "/home/vivek/Downloads/ICC WT20"


if __name__ == "__main__":
    
    files = os.listdir(DATASET_PATH)
    
    for file in files:
        
        filepath = os.path.join(DATASET_PATH, file)
        
        #print(filepath)
        
        cap = cv2.VideoCapture(filepath)

        
        features = []

   
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
  
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

              
                results = model(frame, size=640)


                feature = results.pandas().xyxy[0].values.tolist()
                features.append(feature)

            else:
                break
            
            print(features)
            '''

        cap.release()

        features = np.array(features)
        


        pca = PCA(n_components=2)

 
        pca_features = pca.fit_transform(features)

                        features.append(feature)

        kmeans = KMeans(n_clusters=5)

    
        labels = kmeans.fit_predict(pca_features)

        

        plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels)

    
        plt.show()
'''
