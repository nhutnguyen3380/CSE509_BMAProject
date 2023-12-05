import cv2
import numpy as np
from math import log10, sqrt

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def three_step_search(prev_frame, curr_frame, block_size=16):
    height, width = prev_frame.shape[:2]
    motion_vectors = []

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            min_mse = float('inf')
            best_x, best_y = 0, 0

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    search_y = y + dy
                    search_x = x + dx

                    if search_y < 0 or search_y + block_size > height or search_x < 0 or search_x + block_size > width:
                        continue

                    block_prev = prev_frame[y:y + block_size, x:x + block_size]
                    block_curr = curr_frame[search_y:search_y + block_size, search_x:search_x + block_size]
                    mse = np.mean((block_prev - block_curr) ** 2)

                    if mse < min_mse:
                        min_mse = mse
                        best_x, best_y = dx, dy

            motion_vectors.append((x + best_x, y + best_y))

    return motion_vectors

def main():
    video_path = 'Foreman360p.mp4'
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_psnr = 0.0
    frame_count = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        motion_vectors = three_step_search(prev_frame, curr_frame_gray)

        reconstructed_frame = np.zeros_like(curr_frame)

        for vector in motion_vectors:
            x, y = vector
            reconstructed_frame[y:y + 16, x:x + 16] = curr_frame[y:y + 16, x:x + 16]

        psnr = calculate_psnr(curr_frame, reconstructed_frame)
        total_psnr += psnr
        frame_count += 1

        prev_frame = curr_frame_gray

    average_psnr = total_psnr / frame_count
    print(f'Average PSNR: {average_psnr:.2f} dB')

    cap.release()

if __name__ == "__main__":
    main()
