import cv2
import numpy as np

def separate_images(big_img):
    # Load the large image
    big_img = cv2.imread('bigimage.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image dimensions are as expected
    assert big_img.shape[1] == 5120 and big_img.shape[0] == 800, "Image dimensions do not match expected size"
    
    # Define the width and height of each sub-image
    width = 1280
    height = 800
    
    # Extract each image
    green_img = big_img[0:height, 0:width]
    red_img = big_img[0:height, width:2*width]
    red_edge_img = big_img[0:height, 2*width:3*width]
    nir_img = big_img[0:height, 3*width:4*width]
    
    # Save the separated images
    cv2.imwrite('green.jpg', green_img)
    cv2.imwrite('red.jpg', red_img)
    cv2.imwrite('red_edge.jpg', red_edge_img)
    cv2.imwrite('nir.jpg', nir_img)

def align_images(reference_img_path, target_img_paths):
    # Load the reference image
    reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    keypoints_ref, descriptors_ref = orb.detectAndCompute(reference_img, None)
    
    def align_image(target_img_path):
        # Load the target image
        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Detect keypoints and descriptors in the target image
        keypoints_target, descriptors_target = orb.detectAndCompute(target_img, None)
        
        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_ref, descriptors_target)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched keypoints
        points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_target = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Compute the homography matrix
        M, mask = cv2.findHomography(points_target, points_ref, cv2.RANSAC, 5.0)
        
        # Warp the target image to align with the reference image
        aligned_img = cv2.warpPerspective(target_img, M, (reference_img.shape[1], reference_img.shape[0]))
        
        return aligned_img
    
    # Align each target image with the reference image
    aligned_images = {}
    for target_img_path in target_img_paths:
        aligned_img = align_image(target_img_path)
        aligned_images[target_img_path] = aligned_img
        
    # Save the aligned images
    for img_path, img in aligned_images.items():
        filename = img_path.split('/')[-1].replace('.jpg', '_aligned.jpg')
        cv2.imwrite(filename, img)

# Step 1: Separate the large image into 4 smaller images
separate_images('big_image.jpg')

# Step 2: Align the separated images (red, red edge, and NIR) to the green image
align_images('green.jpg', ['red.jpg', 'red_edge.jpg', 'nir.jpg'])
