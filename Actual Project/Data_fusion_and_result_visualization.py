from imutils import paths
import numpy as np
import imutils
import cv2
import cv2 as cv
import os
import time

# The data format is: (frames, sources, objects, xyz coordinates)
with open('data_from_all_sources_occluded_modified.npy', 'rb') as f:
    data_from_all_sources = np.load(f)

position_of_chessboard = (1.95, 0.81) # The top left corner of the chessboard (In room coordinates)
width, height = 3.27, 2.75            # The dimensions of the part of the room where we operated.
layout_image_original = cv.imread('grid_background.jpg')
pixel_width, pixel_height, _ = layout_image_original.shape
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

# The mapping between the IP address of the R-pi, to the physical understanding of our labeling (for reference purposes only)
# 209: Between W1 and W2 (anchor 2)
# 85: On W1 (anchor 1)
# 28: Between W2 and W3 (anchor 3)
data_from_all_anchors = data_from_all_sources[:, 0:3, :, :]
data_from_all_anchors = np.transpose(data_from_all_anchors, (0, 2, 1, 3))  # Change to (frame_idx, obj_idx, source_idx, xyz)

object_position_relative_to_chessboard = np.zeros((2, 3))           # obj_index, xyz
object_position_relative_to_room = np.zeros((2, 2))                 # obj_index, xy (we don't need the z coordinate, since it is the same everywhere from here on)
object_position_relative_to_image = np.zeros((2, 2), dtype=int)     # obj_index, x & y in pixel values for plotting

last_50_positions = []
for frame_idx, data_point in enumerate(data_from_all_anchors):  # Iterate over frames
    for obj_index, obj_data in enumerate(data_point):
        object_position_relative_to_chessboard[obj_index] = np.mean(obj_data[obj_data[:, 0] > -100], axis=0)
        # Note that the positive x for chessboard is positive y for room, and positive y for chessboard is negative x for room
    
    # Have the positions relative to the chessboard here, note that these are all for the CV side, since sound localization has different needs
    object_position_relative_to_room[:, 0] = ( -1 * object_position_relative_to_chessboard[:, 1] ) + position_of_chessboard[0]  
    object_position_relative_to_room[:, 1] = ( object_position_relative_to_chessboard[:, 0] ) + position_of_chessboard[1]
    
    
    # Fusing the sound localization data if they have picked up (it is already shifted and no need for changing axis)    
    if data_from_all_sources[frame_idx, 3, 0, 0] > -100:
    # Sound localization has valid data to be fused to the x and y position of target 1
        if object_position_relative_to_room[0, 0] > 0: # Cameras (at least one of them) picked it up, fuse the data            
            object_position_relative_to_room[0, :] = (3 * object_position_relative_to_room[0, :] + data_from_all_sources[frame_idx, 3, 0, 0:2] ) / 4
        else:
            # None of the cameras picked up the object (target 1), but the sound localization did, so only use that
            object_position_relative_to_room[0, :] = data_from_all_sources[frame_idx, 3, 0, 0:2]            
        
    if data_from_all_sources[frame_idx, 4, 1, 0] > -100:
        # Sound localization has valid data to be fused to the x and y position of target 2
        if object_position_relative_to_room[1, 0] > 0: # Cameras (at least one of them) picked it up, fuse the data    
            object_position_relative_to_room[1, :] = (3 * object_position_relative_to_room[1, :] + data_from_all_sources[frame_idx, 4, 1, 0:2] ) / 4
        else:
            # None of the cameras picked up the object (target 2), but the sound localization did, so only use that
            object_position_relative_to_room[1, :] = data_from_all_sources[frame_idx, 4, 1, 0:2]
            
    object_position_relative_to_image[:, 0] = np.array((object_position_relative_to_room[:, 0] / width * pixel_width), dtype=int)
    object_position_relative_to_image[:, 1] = np.array((object_position_relative_to_room[:, 1] / height * pixel_height), dtype=int)

    object_positions = [(object_position_relative_to_image[0, 0], object_position_relative_to_image[0, 1]), (object_position_relative_to_image[1, 0], object_position_relative_to_image[1, 1])]
    if len(last_50_positions) > 50:
        last_50_positions.pop(0)
    last_50_positions.append(object_positions)
    
    # We used the last 50 positions simply for demonstration of movement, it is not necessary by any means
    temp_image = layout_image_original.copy()
    for idx, position_of_objects in enumerate(last_50_positions):
        radius = int(6 * idx / 50)
        color = (0, 255, 0)
        thickness = -1
        cv2.circle(temp_image, position_of_objects[0], radius, color, thickness)
        cv2.circle(temp_image, position_of_objects[1], radius, color, thickness)

    # When plotting the final visualization, the x and y coordinate is represented via the position in the room, and we print the z coordinate (in terms of cm)
    cv2.putText(temp_image, '1, ' + str(object_position_relative_to_chessboard[0, 2])[3:5], last_50_positions[-1][0], cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    cv2.putText(temp_image, '2, ' + str(object_position_relative_to_chessboard[1, 2])[3:5], last_50_positions[-1][1], cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    cv.imshow('output', temp_image)
    # Have the waitkey wait for a duration such that it syncs up well with the video
    k = cv2.waitKey(270)        
    if k%256 == 27:   # Stops when ESC is entered
        break
cv2.destroyAllWindows()