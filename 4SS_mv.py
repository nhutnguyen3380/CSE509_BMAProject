import cv2
import numpy as np


def motion_estimation_4step_search(imgP, imgI, mbSize, p):
    height, width = imgP.shape[:2]
    mot_vectors = np.zeros((height // mbSize, width // mbSize, 2))

    for y in range(0, height, mbSize):
        for x in range(0, width, mbSize):
            min_mad = float('inf')
            best_mv = (0, 0)

            # first step, check all 9 points to get a rough idea of the vectors in the border region
            for mv_y in range(-p, p + 1, p):
                for mv_x in range(-p, p + 1, p):
                    mad = calculate_mad(imgP, imgI, x, y, mv_x, mv_y, mbSize)
                    if mad < min_mad:
                        min_mad = mad
                        best_mv = (mv_x, mv_y)

            # every other step narrows the search size using the minimum point found in the previous step
            for step in range(int(np.log2(p))):
                step_size = 2**step
                for mv_y in range(best_mv[1] - step_size, best_mv[1] + step_size + 1, step_size):
                    for mv_x in range(best_mv[0] - step_size, best_mv[0] + step_size + 1, step_size):
                        mad = calculate_mad(imgP, imgI, x, y, mv_x, mv_y, mbSize)
                        if mad < min_mad:
                            min_mad = mad
                            best_mv = (mv_x, mv_y)

            # at the end store the final minimum cost found in the eight points marked by '4'
            mot_vectors[y // mbSize, x // mbSize] = best_mv

    return mot_vectors


# helper function to find the mean absolute difference without stepping out of bounds
def calculate_mad(imgP, imgI, x, y, mv_x, mv_y, mb_size):
    block = imgI[y:y + mb_size, x:x + mb_size]

    target_y, target_x = y + mv_y, x + mv_x
    if 0 <= target_y < imgP.shape[0] - mb_size and 0 <= target_x < imgP.shape[1] - mb_size:
        target_block = imgP[target_y:target_y + mb_size, target_x:target_x + mb_size]
        return np.sum(np.abs(block - target_block))
    else:
        return float('inf')


# function to take the motion vectors and using the cv2 arrow function to draw it over the image
def draw_colored_vectors(current_frame, motion_vectors, block_size):
    for y in range(0, current_frame.shape[0] - block_size + 1, block_size):
        for x in range(0, current_frame.shape[1] - block_size + 1, block_size):
            mv_x, mv_y = motion_vectors[y // block_size, x // block_size]

            # Calculate the endpoint of the arrow
            start_point = (x + block_size // 2, y + block_size // 2)
            end_point = (int(start_point[0] + mv_x), int(start_point[1] + mv_y))

            # Draw arrow on the frame with colored vectors
            cv2.arrowedLine(current_frame, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)


# load any one of the videos
video_path = 'carPOV.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# prepare the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_4step_search.mp4', fourcc, fps, (width, height))

# the parameters for motion estimation, change these to change the size of the macro blocks or search size
# though improper values will lead to out of bound errors or poor results
block_size = 16
search_region = 8

# read the first frame
ret, previous_frame = cap.read()
previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# while there are more frames to be read, keep reading and drawing motion vectors on them
while ret:
    ret, current_frame = cap.read()
    if not ret:
        break

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # calculate motion vectors using 4-step search
    motion_vectors = motion_estimation_4step_search(current_frame_gray, previous_frame_gray, block_size, search_region)

    # draw colored motion vectors on the current frame
    draw_colored_vectors(current_frame, motion_vectors, block_size)

    # put the colored frame in the output
    out.write(current_frame)

    # our current frame will be our next previous frame
    previous_frame_gray = current_frame_gray

# release resources
cap.release()
out.release()
