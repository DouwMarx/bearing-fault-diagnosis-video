import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm

# Apply a polar coordinate transform to the frames
# Load the center locations .mat file
center_locations = sio.loadmat("Data_rpm_variable.mat", squeeze_me=True, struct_as_record=False)
center  = center_locations["circles"].centers.mean(axis=0)

# Load the .avi files
video = cv2.VideoCapture("rpm_variable_fixed.avi")  # Removed invalid frames
height, width, layers =  video.read()[1].shape
print("Video dimensions before transformation: ", width, height)
video = cv2.VideoCapture("rpm_variable_fixed.avi")  # Removed invalid frames

max_radius = np.min([center[0], center[1], width - center[0], height - center[1]]) # Maximum radius that fits in the image

# Extract the first 100 frames from the video
percentage_channels_to_retain = 0.1
percentage_space_to_retain = 1 #0.5
frames_transformed = []
# num_frames = 1000
# for i in tqdm(range(num_frames)):
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Keep only one channel
    frame = frame[:,:,0]

    # Do a polar transformation of the frame
    frame_polar = cv2.linearPolar(frame, center, max_radius, cv2.WARP_FILL_OUTLIERS,)
    # Discard 5% of the frames near the center
    frame_polar = frame_polar[int(0.05 * max_radius):,:]

    # Reduce the dimensions over columns through interpolation
    if percentage_channels_to_retain < 1 or percentage_space_to_retain < 1:
        frame_polar = cv2.resize(frame_polar, (int(percentage_channels_to_retain * frame_polar.shape[0]),int(percentage_space_to_retain*frame_polar.shape[1])), interpolation=cv2.INTER_AREA)

    frames_transformed.append(frame_polar)



# print("Dimensions of the untransformed frames: ", frame.shape)
print("Dimensions of the polar transformed frames: ", frame_polar.shape)


# Save the polar frames as a numpy array
frames_polar_array = np.array(frames_transformed)
print("Dimensions of the numpy  polar frames array: ", frames_polar_array.shape)
np.save("rpm_variable_{}_frames_{}_channels_{}_space_polar.npy".format(len(frames_transformed),percentage_channels_to_retain,percentage_space_to_retain), frames_polar_array)

# Write out the first 1000 frames as a video
height, width = frames_polar_array[0].shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("example_polar_channel_reduce_{}.avi".format( percentage_channels_to_retain), fourcc, 30, (width, height))
for frame in frames_polar_array[0:100]:
    # Use the same frame for all three channels
    frame_stack = np.stack((frame, frame, frame), axis=2)
    writer.write(frame_stack)
writer.release()





# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter("rpm_variable_100_frames.avi", fourcc, 30, (width, height))
# for frame in frames:
#     video_writer.write(frame)
# video_writer.release()
