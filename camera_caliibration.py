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
image_dir_lens1 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam0'
image_dir_lens2 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam1'
image_dir_lens3 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam2'
image_dir_lens4 = r'C:\Users\manal\Documents\Python\Camera_Callibration\checkerboard_Images\Cam3'

# Load images from each lens
images_lens1 = glob.glob(os.path.join(image_dir_lens1, '*.jpg'))
images_lens2 = glob.glob(os.path.join(image_dir_lens2, '*.jpg'))
images_lens3 = glob.glob(os.path.join(image_dir_lens3, '*.jpg'))
images_lens4 = glob.glob(os.path.join(image_dir_lens4, '*.jpg'))

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
