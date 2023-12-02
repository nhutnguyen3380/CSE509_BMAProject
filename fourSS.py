import cv2
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def motion_estimation_4step_search(imgP, imgI, mbSize, p):
    height, width = imgP.shape[:2]
    motion_vectors = np.zeros((height // mbSize, width // mbSize, 2))

    for y in range(0, height, mbSize):
        for x in range(0, width, mbSize):
            min_mad = float('inf')
            best_mv = (0, 0)

            # Initial step
            for mv_y in range(-p, p + 1, p):
                for mv_x in range(-p, p + 1, p):
                    mad = calculate_mad(imgP, imgI, x, y, mv_x, mv_y, mbSize)
                    if mad < min_mad:
                        min_mad = mad
                        best_mv = (mv_x, mv_y)

            # Subsequent steps with smaller search area
            for step in range(int(np.log2(p))):
                step_size = 2**step
                for mv_y in range(best_mv[1] - step_size, best_mv[1] + step_size + 1, step_size):
                    for mv_x in range(best_mv[0] - step_size, best_mv[0] + step_size + 1, step_size):
                        mad = calculate_mad(imgP, imgI, x, y, mv_x, mv_y, mbSize)
                        if mad < min_mad:
                            min_mad = mad
                            best_mv = (mv_x, mv_y)

            motion_vectors[y // mbSize, x // mbSize] = best_mv

    return motion_vectors

def calculate_mad(imgP, imgI, x, y, mv_x, mv_y, mbSize):
    block = imgI[y:y + mbSize, x:x + mbSize]

    target_y, target_x = y + mv_y, x + mv_x
    if 0 <= target_y < imgP.shape[0] - mbSize and 0 <= target_x < imgP.shape[1] - mbSize:
        target_block = imgP[target_y:target_y + mbSize, target_x:target_x + mbSize]
        return np.sum(np.abs(block - target_block))
    else:
        return float('inf')

def reconstruct_frame(imgP, motion_vectors, mbSize):
    height, width = imgP.shape[:2]
    reconstructed_frame = np.zeros_like(imgP, dtype=np.uint8)

    for y in range(0, height, mbSize):
        for x in range(0, width, mbSize):
            mv_x, mv_y = motion_vectors[y // mbSize, x // mbSize]

            mv_x, mv_y = int(mv_x), int(mv_y)

            block = imgP[y + mv_y:y + mv_y + mbSize, x + mv_x:x + mv_x + mbSize]

            reconstructed_frame[y:y + mbSize, x:x + mbSize] = block

    return reconstructed_frame

def compute_metrics(frame_original, frame_reconstructed):
    mse = mean_squared_error(frame_original, frame_reconstructed)
    psnr = peak_signal_noise_ratio(frame_original, frame_reconstructed)
    ssim, _ = structural_similarity(frame_original, frame_reconstructed, full=True)

    return mse, psnr, ssim

# Load the video
video_path = "Foreman360p.mp4"
cap = cv2.VideoCapture(video_path)

# Number of frames to examine
num_frames_to_examine = 5

# Parameters for motion estimation
mbSize = 16
p = 8

# Initialize variables for average metrics
average_mse = 0
average_psnr = 0
average_ssim = 0

# Loop through consecutive frames, compute metrics, and display reconstructed frames
for i in range(num_frames_to_examine - 1):
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    motion_vectors = motion_estimation_4step_search(frame1_gray, frame2_gray, mbSize, p)
    reconstructed_frame = reconstruct_frame(frame1_gray, motion_vectors, mbSize)

    metrics = compute_metrics(frame2_gray, reconstructed_frame)
    average_mse += metrics[0]
    average_psnr += metrics[1]
    average_ssim += metrics[2]

    # Display the reconstructed frame with window title
    window_title = f"Frame {i + 1} - Frame {i + 2}"
    cv2.imshow(window_title, reconstructed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Calculate average metrics
average_mse /= num_frames_to_examine - 1
average_psnr /= num_frames_to_examine - 1
average_ssim /= num_frames_to_examine - 1

# Display the average metrics
print(f"Average MSE: {average_mse}")
print(f"Average PSNR: {average_psnr} dB")
print(f"Average SSIM: {average_ssim}")

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
