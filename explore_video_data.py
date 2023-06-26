# Load the .avi files and extract the frames from the videos

import os
from datetime import time
import plotly.graph_objects as go
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Load the .avi files
video = cv2.VideoCapture("rpm_variable.avi")

# Extract the first 100 frames from the video
num_frames = 1000
frames = []
for i in range(num_frames):
    ret, frame = video.read()
    if ret:
        frames.append(frame)
    else:
        break

# # Save the first 100 frames as a new video
height, width, layers = frames[0].shape
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter("rpm_variable_100_frames.avi", fourcc, 30, (width, height))
# for frame in frames:
#     video_writer.write(frame)
# video_writer.release()

print("Dimensions of the untransformed frames: ", frames[0].shape)

# Apply a polar coordinate transform to the frames
# Load the center locations .mat file
center_locations = sio.loadmat("Data_rpm_variable.mat", squeeze_me=True, struct_as_record=False)
center  = center_locations["circles"].centers.mean(axis=0)

max_radius = np.min([center[0], center[1], width - center[0], height - center[1]]) # Maximum radius that fits in the image
frames_polar = []
percentage_channels_to_retain = 1
for frame in tqdm(frames):
    # Reduce the resolution of the frames along the radius by setting the size
    frame_polar = cv2.linearPolar(frame, center, max_radius, cv2.WARP_FILL_OUTLIERS,)
    # Discard 10% of the frames near the center
    # frame_polar = frame_polar[:, int(0.2 * max_radius):]

    # Reduce the dimensions over columns through interpolation
    if percentage_channels_to_retain < 1:
        frame_polar = cv2.resize(frame_polar, (int(percentage_channels_to_retain * frame_polar.shape[0]),frame_polar.shape[1]), interpolation=cv2.INTER_AREA)

    frames_polar.append(frame_polar)


print("Dimensions of the polar transformed frames: ", frames_polar[0].shape)

# Extract the red, green, and blue channels from the first frame
red_channel = frames_polar[0][:, :, 2]
green_channel = frames_polar[0][:, :, 1]
blue_channel = frames_polar[0][:, :, 0]

# Check if the red, green, and blue channels are identical
print("Are the red, green, and blue channels identical? ", np.all(red_channel == green_channel) and np.all(green_channel == blue_channel))

# Save the polar frames as a numpy array, keeping only one channel (num_frames, height, width)
frames_polar_array = np.array(frames_polar)[:, :, :, 0]
np.save("rpm_variable_{}_frames_{}_channels_polar.npy".format(num_frames,percentage_channels_to_retain), frames_polar_array)





# Save the polar transformed frames as a new video
# height, width, layers = frames_polar[0].shape
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter("rpm_variable_100_frames_polar.avi", fourcc, 30, (width, height))
# for frame in frames_polar:
#     video_writer.write(frame)
# video_writer.release()
