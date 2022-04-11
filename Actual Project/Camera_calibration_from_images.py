import numpy as np
import cv2 as cv
import glob
import yaml
square_size = 0.054 # You can tune this. This size reflects the distance between the corner points of the squares 
image_size = (640, 480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Horizontal and vertical are flipped, but lazy. The numbers must be exact, and they are the corners inside, not outside
horizontal = 4 
vertical = 7
objp = np.zeros((horizontal*vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:horizontal,0:vertical].T.reshape(-1,2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./Calibration_Images/*.png')
success_count = 0
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (horizontal,vertical), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners, totally optional, simply for visualization
        #cv.drawChessboardCorners(img, (horizontal,vertical), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(200)
        success_count += 1
cv.destroyAllWindows()
print('Successfully used ' + str(success_count) + ' images.')
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
if ret:
    np.save('intrinsic_matrix', mtx)
    np.save('distortion_vector', dist)
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, mtx, image_size, cv.CV_32FC1)
    np.save('undistortion_map_x', mapx)
    np.save('undistortion_map_y', mapy)
print('Calibration Complete')