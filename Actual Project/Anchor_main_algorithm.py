from imutils import paths
import numpy as np
import imutils
import cv2
import cv2 as cv
import os
import glob
import pyzbar
import pyzbar.pyzbar as pyzbar
import time

from packfunctions import *
import socket
import struct
import sys 

#Setup
s_COMMAND = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s_COMMAND.bind(ANCHOR_3_IN)
s_COMMAND.settimeout(None)
# Load up the calibration results and undistortion maps
mtx = np.load('intrinsic_matrix.npy')
dist = np.load('distortion_vector.npy')
mapx = np.load('undistortion_map_x.npy')
mapy = np.load('undistortion_map_y.npy')


def retrieve_correcting_elements():
    square_size = 0.054

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Flipping horizontal and verical, so we don't need to switch cols
    horizontal = 4 
    vertical = 7
    objp = np.zeros((horizontal*vertical,3), np.float32)
    objp[:,:2] = np.mgrid[0:horizontal,0:vertical].T.reshape(-1,2) * square_size
    
    img = cv.imread('calibration_image_0.png')    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (horizontal,vertical), None)
    # If found, add object points, image points (after refining them)
    
    if ret == True:
        ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)
    
        correcting_rotation_matrix, _ = cv2.Rodrigues(rvecs)
        correcting_translation_vector = tvecs.T
        print('Extracted correcting features for the anchor')
        return correcting_rotation_matrix, correcting_translation_vector
    else:
        print('Could not find the checkerboard')

correcting_rotation_matrix, correcting_translation_vector = retrieve_correcting_elements()

def change_coordinate_system(x, y, z):
    x -= correcting_translation_vector[0, 0]
    y -= correcting_translation_vector[0, 1]
    z -= correcting_translation_vector[0, 2]
    row1 = x * correcting_rotation_matrix[0, 0] + y * correcting_rotation_matrix[1, 0] + z * correcting_rotation_matrix[2, 0]
    row2 = x * correcting_rotation_matrix[0, 1] + y * correcting_rotation_matrix[1, 1] + z * correcting_rotation_matrix[2, 1]
    row3 = x * correcting_rotation_matrix[0, 2] + y * correcting_rotation_matrix[1, 2] + z * correcting_rotation_matrix[2, 2]
    x = row1
    y = row2
    z = row3
    return np.array([x, y, z]).T



while time.time() < 1649005410:
    print(time.time())

image_size = (1280, 720) # Provides a good balance between performance and FPS. 480p: 0.0465s, 720p: 0.145s
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

objp = np.zeros((2*2,3), np.float32)   # The order is top left, top right, bottom left, bottom right. img points must match
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2) * 0.4   # Tune this value by checking with IRL measurements, was 0.245 for A4, 0.4 for our QR code

# data_points = []
while True:
    ret, frame = cam.read()
    if not ret:
        print("End of Video/Coudln't grab frames")
        break
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
 
    decodedObjects = pyzbar.decode(frame)
    data_at_frame = -100.0 * np.ones((2, 3))   # First dimensions is for objects, second is for x and y coordinate of that object
    for idx, decodedObject in enumerate(decodedObjects):
        points = decodedObject.polygon
        for j in range(4):
            cv2.line(frame, points[j], points[((j+1)%4)], (255,0,0), 3)
        points = np.array(decodedObject.polygon, dtype=int)
        x = 0
        y = 0
        z = 0
        for i in range(4):        
            temp = [[[points[(0 + i)%4, 0], points[(0 + i)%4, 1]]], [[points[(3 + i)%4, 0], points[(3 + i)%4, 1]]], [[points[(1 + i)%4, 0], points[(1 + i)%4, 1]]], [[points[(2 + i)%4, 0], points[(2 + i)%4, 1]]]]
            temp = np.array(temp, dtype=np.float32)
            ret,rvecs, tvecs = cv.solvePnP(objp, temp, mtx, dist)
            x += tvecs[0]/4
            y += tvecs[1]/4
            z += tvecs[2]/4
        global_coordinates = change_coordinate_system(x, y, z)        
        
        
        cv2.putText(frame, str(global_coordinates) + ' Data: ' + (decodedObject.data).decode('utf-8'), (20, (idx + 1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # data_at_frame[int((decodedObject.data).decode('utf-8'))-1, :] = global_coordinates
    
    #################BEGIN NETWORK TRANSMISSION#############################
    packed_data = packSF452data('d3p',data_at_frame.reshape(-1),'a3')
    recvSF452data(s_COMMAND)
    sendSF452data(s_COMMAND,packed_data,TEST_SERVER_IN)
    #################END NETWORK TRANSMISSION###############################
    
    
    # data_points.append(data_at_frame)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("output", frame)   
    k = cv2.waitKey(1)
    if k%256 == 27:   # Stops when ESC is entered
        break
    # elif k%256 == 32: # Spacebar to skip a few frames
    #     for i in range(200):
    #         cam.read()
cam.release()
cv2.destroyAllWindows()
# data_points = np.array(data_points)
# with open('anchor_data.npy', 'wb') as f:
#     np.save(f, data_points)


