# video_path = '/home/vivek/VisionWorkspace/dataset/ICC WT20/ICC WT20 - Afghanistan vs South Africa - Match Highlights.avi'

import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Path to the video file
video_path = '/home/vivek/VisionWorkspace/dataset/ICC WT20/ICC WT20 - Afghanistan vs South Africa - Match Highlights.avi'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Initialize an empty list to store feature vectors
feature_vectors = []

while True:
    # Capture two consecutive frames from the video
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if not ret:
        break

    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Dense Pyramid Lucas-Kanade
    p1 = cv2.goodFeaturesToTrack(frame1_gray, 500, 0.01, 10)
    p2, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, p1, None, **lk_params)

    # Select good points
    good_new = p2[st == 1]
    good_old = p1[st == 1]

    # Compute the feature vectors for the optical flow lines
    feature_vectors.append([])
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        center = (int(a), int(b))
        feature_vectors[-1].append(center)

    # Draw optical flow lines on the first frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        center = (int(a), int(b))
        frame1 = cv2.circle(frame1, center, 5, (0, 255, 0), -1)
        frame1 = cv2.line(frame1, center, (int(c), int(d)), (0, 0, 255), 2)

    # Display the first frame with optical flow lines
    cv2.imshow('Dense Pyramid Lucas-Kanade Optical Flow', frame1)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Compute the distance matrix between feature vectors using DTW and FastDTW
distance_matrix = np.zeros((len(feature_vectors), len(feature_vectors)))
for i in range(len(feature_vectors)):
    if i == 50:  # limit feature vectors to 5
        break
    for j in range(i + 1, len(feature_vectors)):
        if j == 500:  # limit feature vectors to 5
            break
        # Compute the distance between the two feature vectors using DTW
        distance, _ = fastdtw(feature_vectors[i], feature_vectors[j], dist=euclidean)
        distance_matrix[i, j] = distance_matrix[j, i] = distance
        print(f"DTW distance between feature vector {i} and feature vector {j}: {distance}")

# Concatenate feature vectors into a single array
feature_array = np.concatenate(feature_vectors)

# Apply spectral clustering to group the feature vectors
n_clusters = 5
spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
labels = spectral.fit_predict(distance_matrix)

# Plot the output of the spectral clustering algorithm
plt.scatter(feature_array[:, 0], feature_array[:, 1], c=labels)
plt.title('Spectral Clustering Output')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()