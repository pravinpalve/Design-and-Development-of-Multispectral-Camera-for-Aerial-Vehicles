
'''
WORKING CODE TO CALCULATE CALLIBRATION MATRIX AND DISTORTION COEFFICIENTS AND SAVE THEM IN THE RESPECTIVE FOLDERS
'''
# YOU NEED TO CHANGE FOLDER PATH AT TWO PLACES

import cv2
import numpy as np
import glob
import os

# Termination criteria for corner subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,9,0) for a 9x10 board
objp = np.zeros((9*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:10].T.reshape(-1, 2)

# Arrays to store object points and image points for all four lenses
objpoints = []  # 3d point in real world space
imgpoints_lens1 = []  # 2d points in image plane for lens 1
imgpoints_lens2 = []  # 2d points in image plane for lens 2
imgpoints_lens3 = []  # 2d points in image plane for lens 3
imgpoints_lens4 = []  # 2d points in image plane for lens 4

# Directory containing the calibration images for each lens
### Change the folder path here###
image_dir_lens1 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam0'
image_dir_lens2 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam1'
image_dir_lens3 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam2'
image_dir_lens4 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam3'

# Load images from each lens
images_lens1 = glob.glob(os.path.join(image_dir_lens1, '*.jpg'))
images_lens2 = glob.glob(os.path.join(image_dir_lens2, '*.jpg'))
images_lens3 = glob.glob(os.path.join(image_dir_lens3, '*.jpg'))
images_lens4 = glob.glob(os.path.join(image_dir_lens4, '*.jpg'))

# Process images for each lens
for fname1, fname2, fname3, fname4 in zip(images_lens1, images_lens2, images_lens3, images_lens4):
    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)
    img3 = cv2.imread(fname3)
    img4 = cv2.imread(fname4)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, (9, 10), None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, (9, 10), None)
    ret3, corners3 = cv2.findChessboardCorners(gray3, (9, 10), None)
    ret4, corners4 = cv2.findChessboardCorners(gray4, (9, 10), None)
    
    # If found, add object points, image points (after refining them)
    if ret1 and ret2 and ret3 and ret4:
        objpoints.append(objp)
        
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        imgpoints_lens1.append(corners1)
        
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints_lens2.append(corners2)
        
        corners3 = cv2.cornerSubPix(gray3, corners3, (11, 11), (-1, -1), criteria)
        imgpoints_lens3.append(corners3)
        
        corners4 = cv2.cornerSubPix(gray4, corners4, (11, 11), (-1, -1), criteria)
        imgpoints_lens4.append(corners4)

# Calibration for each lens
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_lens1, gray1.shape[::-1], None, None)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_lens2, gray2.shape[::-1], None, None)
ret3, mtx3, dist3, rvecs3, tvecs3 = cv2.calibrateCamera(objpoints, imgpoints_lens3, gray3.shape[::-1], None, None)
ret4, mtx4, dist4, rvecs4, tvecs4 = cv2.calibrateCamera(objpoints, imgpoints_lens4, gray4.shape[::-1], None, None)

# Saving calibration data for each lens
np.savez('calibration_data_lens1.npz', ret=ret1, mtx=mtx1, dist=dist1, rvecs=rvecs1, tvecs=tvecs1)
np.savez('calibration_data_lens2.npz', ret=ret2, mtx=mtx2, dist=dist2, rvecs=rvecs2, tvecs=tvecs2)
np.savez('calibration_data_lens3.npz', ret=ret3, mtx=mtx3, dist=dist3, rvecs=rvecs3, tvecs=tvecs3)
np.savez('calibration_data_lens4.npz', ret=ret4, mtx=mtx4, dist=dist4, rvecs=rvecs4, tvecs=tvecs4)

print("Calibration completed for all lenses")
print("\n\n Undistorting the images and saving it in amera callibrated_images folder\n\n")



# Function to undistort an image using calibration data
def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # Crop the image based on the roi
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    return undistorted_img


# Load calibration data
calibration_data_lens1 = np.load('calibration_data_lens1.npz')
calibration_data_lens2 = np.load('calibration_data_lens2.npz')
calibration_data_lens3 = np.load('calibration_data_lens3.npz')
calibration_data_lens4 = np.load('calibration_data_lens4.npz')

mtx1 = calibration_data_lens1['mtx']
dist1 = calibration_data_lens1['dist']
mtx2 = calibration_data_lens2['mtx']
dist2 = calibration_data_lens2['dist']
mtx3 = calibration_data_lens3['mtx']
dist3 = calibration_data_lens3['dist']
mtx4 = calibration_data_lens4['mtx']
dist4 = calibration_data_lens4['dist']

# Directories to save undistorted images
### Change folder path here ###
undistorted_image_dir = r'C:\Users\manal\Documents\Python\Camera_Callibration\callibrated_images'
os.makedirs(undistorted_image_dir, exist_ok=True)

undistorted_dir_lens1 = os.path.join(undistorted_image_dir, 'Cam0')
undistorted_dir_lens2 = os.path.join(undistorted_image_dir, 'Cam1')
undistorted_dir_lens3 = os.path.join(undistorted_image_dir, 'Cam2')
undistorted_dir_lens4 = os.path.join(undistorted_image_dir, 'Cam3')

os.makedirs(undistorted_dir_lens1, exist_ok=True)
os.makedirs(undistorted_dir_lens2, exist_ok=True)
os.makedirs(undistorted_dir_lens3, exist_ok=True)
os.makedirs(undistorted_dir_lens4, exist_ok=True)









# Process and undistort images for each lens
for fname1, fname2, fname3, fname4 in zip(images_lens1, images_lens2, images_lens3, images_lens4):
    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)
    img3 = cv2.imread(fname3)
    img4 = cv2.imread(fname4)
    
    undistorted_img1 = undistort_image(img1, mtx1, dist1)
    undistorted_img2 = undistort_image(img2, mtx2, dist2)
    undistorted_img3 = undistort_image(img3, mtx3, dist3)
    undistorted_img4 = undistort_image(img4, mtx4, dist4)
    
    # Get the base filenames and save undistorted images
    base_name1 = os.path.basename(fname1)
    base_name2 = os.path.basename(fname2)
    base_name3 = os.path.basename(fname3)
    base_name4 = os.path.basename(fname4)
    
    cv2.imwrite(os.path.join(undistorted_dir_lens1, base_name1), undistorted_img1)
    cv2.imwrite(os.path.join(undistorted_dir_lens2, base_name2), undistorted_img2)
    cv2.imwrite(os.path.join(undistorted_dir_lens3, base_name3), undistorted_img3)
    cv2.imwrite(os.path.join(undistorted_dir_lens4, base_name4), undistorted_img4)

print("Undistortion completed and images saved in respective folders.")
