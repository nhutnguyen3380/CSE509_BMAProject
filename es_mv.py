import cv2
import numpy as np


def motion_estimation_exhaustive_search(imgP, imgI, mbSize, p):
    height, width = imgP.shape[:2]

    # find the number of macroblocks that fit in the image, so that we don't go out of bounds
    num_mb_height = height // mbSize
    num_mb_width = width // mbSize

    # prep a numpy array of zeroes to contain the best vectors for this pair of frames
    final_mv = np.zeros((num_mb_height, num_mb_width, 2))

    for y_mb in range(num_mb_height):
        for x_mb in range(num_mb_width):
            y = y_mb * mbSize
            x = x_mb * mbSize

            # we're looking for the coords of the block with the lowest cost in the neighborhood, note neighborhood
            # is the entire rest of the image in ES
            min_mad = float('inf')
            best_mv = (0, 0)

            # search for the cost in (2p +1) blocks vertically and horizontally, remember p is our search parameter
            for mv_y in range(-p, p + 1):
                for mv_x in range(-p, p + 1):
                    # create our current block
                    block = imgI[y:y + mbSize, x:x + mbSize]

                    # check to make sure the motion vectors are within the target frame boundaries
                    if 0 <= y + mv_y < height - mbSize and 0 <= x + mv_x < width - mbSize:
                        target_block = imgP[y + mv_y:y + mv_y + mbSize, x + mv_x:x + mv_x + mbSize]

                        # calculate the mean absolute difference (MAD) between the blocks and see if its the smallest
                        mad = np.sum(np.abs(block - target_block))

                        # if this is the smallest cost so far, save its coords and go on
                        if mad < min_mad:
                            min_mad = mad
                            best_mv = (mv_x, mv_y)

            # store the best motion vector for the current macroblock to return later
            final_mv[y_mb, x_mb] = best_mv

    return final_mv


video_path = 'Foreman360p.mp4'
#video_path = 'meerkat.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)
ret, previous_frame = cap.read()

# Convert the frame to grayscale
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Parameters
block_size = 16
search_region = 8

# Get video properties for the output video
fps = cap.get(cv2.CAP_PROP_FPS)
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_ES_motion_vectors.mp4', fourcc, fps, (width, height))

while ret:
    ret, current_frame = cap.read()
    if not ret:
        break

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate motion vectors
    motion_vectors = motion_estimation_exhaustive_search(current_frame_gray, previous_frame, block_size, search_region)

    # Overlaying motion vectors
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            mv_x, mv_y = motion_vectors[y // block_size, x // block_size]

            # Calculate the endpoint of the arrow
            start_point = (x + block_size // 2, y + block_size // 2)
            end_point = (int(start_point[0] + mv_x), int(start_point[1] + mv_y))

            # Draw arrow on the frame
            cv2.arrowedLine(current_frame, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)

    # Write frame with motion vectors to the output file
    out.write(current_frame)

    previous_frame = current_frame_gray

# Release resources
cap.release()
out.release()