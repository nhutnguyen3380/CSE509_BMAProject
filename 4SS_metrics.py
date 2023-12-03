import cv2
import numpy as np
import time
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


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


# function to reconstruct a frame and return it
def reconstruct_frame(imgP, motion_vectors, mbSize):
    height, width = imgP.shape[:2]
    recon_frame = np.zeros_like(imgP, dtype=np.uint8)

    for y in range(0, height, mbSize):
        for x in range(0, width, mbSize):
            mv_x, mv_y = motion_vectors[y // mbSize, x // mbSize]
            mv_x, mv_y = int(mv_x), int(mv_y)

            block = imgP[y + mv_y:y + mv_y + mbSize, x + mv_x:x + mv_x + mbSize]
            recon_frame[y:y + mbSize, x:x + mbSize] = block

    return recon_frame


# use the skimage library to compute the metrics between the og and recon frames
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
    motion_vectors = motion_estimation_4step_search(frame1_gray, frame2_gray, mbSize, p)
    end_time = time.time()

    reconstructed_frame = reconstruct_frame(frame1_gray, motion_vectors, mbSize)

    metrics = compute_metrics(frame2_gray, reconstructed_frame)
    avg_mse += metrics[0]
    avg_psnr += metrics[1]
    avg_ssim += metrics[2]
    avg_time += end_time - start_time

    # display the reconstructed frame with window title if the user desires it
    if show_imgs:
        window_title = f"4SS Frame {i + 1} - Frame {i + 2}"
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