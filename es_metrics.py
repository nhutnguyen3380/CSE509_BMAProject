import cv2
import numpy as np
import time
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


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

            # we're looking for the coords of the block with the lowest cost in the neighborhood,
            # note neighborhood is the entire rest of the image in ES
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


def reconstruct_frame(img, m_vectors, mbSize):
    height, width = img.shape[:2]
    recon_frame = np.zeros_like(img, dtype=np.uint8)

    # Calculate the number of macroblocks along height and width
    num_mb_height = height // mbSize
    num_mb_width = width // mbSize

    for y_mb in range(num_mb_height):
        for x_mb in range(num_mb_width):
            y = y_mb * mbSize
            x = x_mb * mbSize

            mv_x, mv_y = m_vectors[y_mb, x_mb]

            # Ensure the motion vectors are integers
            mv_x, mv_y = int(mv_x), int(mv_y)

            # Apply motion vector to reconstruct the block
            block = img[y + mv_y:y + mv_y + mbSize, x + mv_x:x + mv_x + mbSize]

            # Place the reconstructed block in the frame
            recon_frame[y:y + mbSize, x:x + mbSize] = block

    return recon_frame


def compute_metrics(frame_original, frame_reconstructed):
    mse = mean_squared_error(frame_original, frame_reconstructed)
    psnr = peak_signal_noise_ratio(frame_original, frame_reconstructed)
    ssim, _ = structural_similarity(frame_original, frame_reconstructed, full=True)

    return mse, psnr, ssim


# load any one of the videos
video_path = "Foreman360p.mp4"
cap = cv2.VideoCapture(video_path)
# if you'd like to have each of the reconstructed frames displayed, set this bool to True
show_imgs = False

# print the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total Frames in Video: {total_frames}")

# Choose the number of frames you'd like to examine
# note that if you want to examine all of them, set it to -1 as a shortcut
num_frames_to_examine = -1
if num_frames_to_examine == -1:
    print(f"Total Frames being evaluated: {total_frames}")
else:
    print(f"Total Frames being evaluated: {num_frames_to_examine}")

# the parameters for motion estimation, change these to change the size of the macro blocks or search size
# though improper values will lead to out of bound errors or poor results
mbSize = 16
p = 8

# initialize variables for average metrics and time
avg_mse = 0
avg_psnr = 0
avg_ssim = 0
avg_time = 0
tot_frames_checked = 0

# loop through the desired consecutive frames, compute metrics, and display reconstructed frames
for i in range(num_frames_to_examine - 1) if num_frames_to_examine != -1 else range(int(total_frames/2)):
    ret1, frame1 = cap.read()

    if not ret1:
        # check if it's the end of the video
        if i == 0:
            print("Failed to read the first frame. Exiting the loop...")
        else:
            print(f"Failed to read frame {i + 1}. Exiting the loop...")
        break

    ret2, frame2 = cap.read()

    # check if the second frame was successfully read
    if not ret2:
        print(f"Failed to read frame {i + 2}. Exiting the loop...")
        break

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    motion_vectors = motion_estimation_exhaustive_search(frame1_gray, frame2_gray, mbSize, p)
    end_time = time.time()

    reconstructed_frame = reconstruct_frame(frame1_gray, motion_vectors, mbSize)

    metrics = compute_metrics(frame2_gray, reconstructed_frame)
    avg_mse += metrics[0]
    avg_psnr += metrics[1]
    avg_ssim += metrics[2]
    avg_time += end_time - start_time

    # display the reconstructed frame with window title if the user desires it
    if show_imgs:
        window_title = f"ES Frame {i + 1} - Frame {i + 2}"
        cv2.imshow(window_title, reconstructed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    tot_frames_checked += 1

# calculate and display rge average metrics and time
avg_mse /= tot_frames_checked
avg_psnr /= tot_frames_checked
avg_ssim /= tot_frames_checked
avg_time /= tot_frames_checked

print(f"Average MSE: {avg_mse}")
print(f"Average PSNR: {avg_psnr} dB")
print(f"Average SSIM: {avg_ssim}")
print(f"Average Time: {avg_time} seconds")

# release the video capture object
cap.release()
cv2.destroyAllWindows()
