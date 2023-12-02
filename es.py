import numpy as np
import cv2


def motion_estimation_exhaustive_search(imgP, imgI, mbSize, p):
    row, col = imgI.shape
    vectors = np.zeros((2, row * col // mbSize ** 2))
    costs = np.ones((2 * p + 1, 2 * p + 1)) * 65537
    computations = 0

    # start from the top left of the image then go block by block each of size = mbSize
    mbCount = 1
    for i in range(0, row - mbSize + 1, mbSize):
        for j in range(0, col - mbSize + 1, mbSize):

            # the search starts here
            # evaluate the cost for (2p + 1) blocks vertically and horizontally, remember p is our search parameter
            for row_i in range(-p, p + 1):
                for col_i in range(-p, p + 1):
                    # row coordinate for current ref block
                    refBlkVer = i + row_i
                    # col coordinate for current ref block
                    refBlkHor = j + col_i

                    if refBlkVer < 0 or refBlkVer + mbSize > row or refBlkHor < 0 or refBlkHor + mbSize > col:
                        continue

                    costs[row_i + p, col_i + p] = cost_func_mad(imgP[i:i + mbSize, j:j + mbSize],
                                                        imgI[refBlkVer:refBlkVer + mbSize,
                                                        refBlkHor:refBlkHor + mbSize],
                                                        mbSize)
                    computations += 1

            # use the min_cost function to find the minimum vector and store it
            dx, dy, min = min_cost(costs)
            # row co-coordinate for the vector
            vectors[0, mbCount - 1] = dy - p - 1
            # col co-coordinate for the vector
            vectors[1, mbCount - 1] = dx - p - 1
            mbCount += 1
            costs = np.ones((2 * p + 1, 2 * p + 1)) * 65537

    motionVect = vectors
    EScomputations = computations / (mbCount - 1)

    return motionVect, EScomputations


# find the mean absolute difference between two images (blocks)
def cost_func_mad(block1, block2, mbSize):
    return np.sum(np.abs(block1 - block2)) / (mbSize ** 2)


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
        motion_vectors, computations = motion_estimation_exhaustive_search(curr_frame_gray, prev_frame_gray, mbSize, p)

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


