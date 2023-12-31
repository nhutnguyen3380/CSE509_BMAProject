# -*- coding: utf-8 -*-
"""overlay.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YpyEIZE33aVM_BfJp4HUCvwkLUsFKY2d
"""

import cv2
import numpy as np

video_path = 'Foreman360p.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)
ret, previous_frame = cap.read()

# Convert the frame to grayscale
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Parameters
block_size = 16
search_region = 2

# Get video properties for the output video
fps = cap.get(cv2.CAP_PROP_FPS)
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_motion_vectors.mp4', fourcc, fps, (width, height))

while ret:
    ret, current_frame = cap.read()
    if not ret:
        break

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate motion vectors
    motion_vectors = ebma_using_matchTemplate(current_frame_gray, previous_frame, block_size, search_region)

    # Overlaying motion vectors
    for (x, y, u, v) in motion_vectors:
        start_point = (int(x), int(y))
        end_point = (int(x + u), int(y + v))
        cv2.arrowedLine(current_frame, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)

    # Write frame with motion vectors to the output file
    out.write(current_frame)

    previous_frame = current_frame_gray

# Release resources
cap.release()
out.release()