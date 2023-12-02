import numpy as np
import cv2
import matplotlib.pyplot as plt


def motion_estimation_4SS(imgP, imgI, mbSize, p):
    row, col = imgI.shape
    vectors = np.zeros((2, row * col // mbSize**2))
    costs = np.ones((3, 3)) * 65537
    computations = 0
    mbCount = 1

    for i in range(0, row - mbSize + 1, mbSize):
        for j in range(0, col - mbSize + 1, mbSize):
            x = j
            y = i
            costs[1, 1] = cost_func_mad(imgP[i:i+mbSize, j:j+mbSize], imgI[i:i+mbSize, j:j+mbSize], mbSize)
            computations += 1

            # first step, evaluate all 16 points
            for row_i in range(-2, 3, 2):
                for col_i in range(-2, 3, 2):
                    refBlkVer = y + row_i
                    refBlkHor = x + col_i
                    if (refBlkVer < 1 or refBlkVer + mbSize - 1 > row or
                            refBlkHor < 1 or refBlkHor + mbSize - 1 > col):
                        continue
                    costRow = row_i // 2 + 1
                    costCol = col_i // 2 + 1
                    if costRow == 1 and costCol == 1:
                        continue
                    costs[costRow, costCol] = cost_func_mad(imgP[i:i+mbSize, j:j+mbSize],
                                                            imgI[refBlkVer:refBlkVer+mbSize, refBlkHor:refBlkHor+mbSize],
                                                            mbSize)
                    computations += 1

            dx, dy, cost = min_cost(costs)

            # assuming the best match is in the center of our search field set the flag = true
            if dx == 1 and dy == 1:
                flag_4ss = 1
            else:
                flag_4ss = 0
                xLast, yLast = x, y
                x = x + (dx - 1) * 2
                y = y + (dy - 1) * 2

            costs = np.ones((3, 3)) * 65537
            costs[1, 1] = cost

            # these nested loops evaluate the best match within a subset of points, starting large then zero-ing in
            stage = 1
            while flag_4ss == 0 and stage <= 2:
                for m in range(-2, 3, 2):
                    for n in range(-2, 3, 2):
                        refBlkVer = y + m
                        refBlkHor = x + n
                        if (refBlkVer < 1 or refBlkVer + mbSize - 1 > row or
                                refBlkHor < 1 or refBlkHor + mbSize - 1 > col):
                            continue
                        if (xLast - 2 <= refBlkHor <= xLast + 2 and
                                yLast - 2 <= refBlkVer <= yLast + 2):
                            continue
                        costRow = m // 2 + 1
                        costCol = n // 2 + 1
                        if costRow == 1 and costCol == 1:
                            continue
                        costs[costRow, costCol] = cost_func_mad(imgP[i:i+mbSize, j:j+mbSize],
                                                                imgI[refBlkVer:refBlkVer+mbSize, refBlkHor:refBlkHor+mbSize],
                                                                mbSize)
                        computations += 1

                dx, dy, cost = min_cost(costs)

                if dx == 1 and dy == 1:
                    flag_4ss = 1
                else:
                    flag_4ss = 0
                    xLast, yLast = x, y
                    x = x + (dx - 1) * 2
                    y = y + (dy - 1) * 2

                costs = np.ones((3, 3)) * 65537
                costs[1, 1] = cost
                stage += 1

            # Final stage
            for m in range(-1, 2):
                for n in range(-1, 2):
                    refBlkVer = y + m
                    refBlkHor = x + n
                    if (refBlkVer < 1 or refBlkVer + mbSize - 1 > row or
                            refBlkHor < 1 or refBlkHor + mbSize - 1 > col):
                        continue
                    costRow = m + 1
                    costCol = n + 1
                    if costRow == 1 and costCol == 1:
                        continue
                    costs[costRow, costCol] = cost_func_mad(imgP[i:i+mbSize, j:j+mbSize],
                                                            imgI[refBlkVer:refBlkVer+mbSize, refBlkHor:refBlkHor+mbSize],
                                                            mbSize)
                    computations += 1

            dx, dy, _ = min_cost(costs)

            x = x + dx - 1
            y = y + dy - 1

            vectors[0, mbCount - 1] = y - i
            vectors[1, mbCount - 1] = x - j
            mbCount += 1

            costs = np.ones((3, 3)) * 65537

    motionVect = vectors
    SS4Computations = computations / (mbCount - 1)

    return motionVect, SS4Computations


# find the mean absolute difference between two images (blocks)
def cost_func_mad(block1, block2, mbSize):
    # Ensure both blocks have the same shape
    min_rows, min_cols = min(block1.shape[0], block2.shape[0]), min(block1.shape[1], block2.shape[1])
    block1 = block1[:min_rows, :min_cols]
    block2 = block2[:min_rows, :min_cols]

    # Calculate the MAD
    return np.sum(np.abs(block1 - block2)) / (min_rows * min_cols)


# in a group of costs, find the smallest one and return its coordinates and value
def min_cost(costs):
    min_index = np.argmin(costs)
    min_row, min_col = divmod(min_index, costs.shape[1])
    return min_col, min_row, costs[min_row, min_col]


def calculate_motion_vectors(video_path, mbSize, p, num_frames=10):
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = cap.get(5)
    total_frames = int(cap.get(7))

    print(f"Video Resolution: {frame_width}x{frame_height}")
    print(f"Frame Rate: {frame_rate} fps")
    print(f"Total Frames: {total_frames}")

    # Initialize variables to store motion vectors
    all_motion_vectors = []
    all_computations = []

    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the first frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Iterate through the specified number of frames
    for _ in range(min(num_frames, total_frames - 1)):
        # Read the next frame
        ret, curr_frame = cap.read()

        # Convert the current frame to grayscale
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate motion vectors using the exhaustive search method
        motion_vectors, computations = motion_estimation_4SS(curr_frame_gray, prev_frame_gray, mbSize, p)

        # Store the motion vectors and computations
        all_motion_vectors.append(motion_vectors)
        all_computations.append(computations)

        # Set the current frame as the previous frame for the next iteration
        prev_frame_gray = curr_frame_gray

    # Release the video capture object
    cap.release()

    return all_motion_vectors, all_computations


def print_motion_vectors(motion_vectors):
    for i, vectors in enumerate(motion_vectors):
        print(f"Frame {i + 1} - Motion Vectors:")
        for j in range(vectors.shape[1]):
            print(f"  Block {j + 1}: ({vectors[0, j]}, {vectors[1, j]})")
        print()


# Example usage
video_path = 'Foreman360p.mp4'
macroblock_size = 16
search_parameter = 7
n_frames = 10

motion_vectors, computations = calculate_motion_vectors(video_path, macroblock_size, search_parameter, n_frames)
print_motion_vectors(motion_vectors)
print(computations)