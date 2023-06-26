import cv2
import numpy as np

# Load the .avi files
video = cv2.VideoCapture("rpm_variable.avi")

height, width, layers =  video.read()[1].shape

# Check if a frame contains NaNs or Infs and replace with the previous frame

fixed_video = cv2.VideoWriter("rpm_variable_fixed.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

prev_frame = np.zeros((height, width))
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Check if more than 10% of the pixels are complete black
    if np.isnan(frame).any() or np.isinf(frame).any() or np.sum(frame == 0) > 0.1 * frame.size:
        frame = prev_frame
    prev_frame = frame
    fixed_video.write(frame)

fixed_video.release()
video.release()

