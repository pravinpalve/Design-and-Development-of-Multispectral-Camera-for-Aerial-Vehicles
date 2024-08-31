import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Description for code:
Image translation for registration
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt

# This function is responsible for translation
def translate_image(image, x_shift, y_shift):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

# Adjusting resolution of images
def match_resolution(images, target_height, target_width, method='pad'):
    matched_images = []
    for img in images:
        if method == 'pad':
            # Calculate padding
            top_pad = (target_height - img.shape[0]) // 2
            bottom_pad = target_height - img.shape[0] - top_pad
            left_pad = (target_width - img.shape[1]) // 2
            right_pad = target_width - img.shape[1] - left_pad
            # Add zero padding
            padded_img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            matched_images.append(padded_img[:target_height, :target_width])
        elif method == 'crop':
            # Center crop to the target size
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            x_start = max(center_x - target_width // 2, 0)
            y_start = max(center_y - target_height // 2, 0)
            cropped_img = img[y_start:y_start + target_height, x_start:x_start + target_width]
            matched_images.append(cropped_img)
    return matched_images

################################################################################################################################

# Load your images (replace with actual paths)
image1 = cv2.imread('/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/ImagesWithFilters/Individual/Cam0/0.jpg')
image2 = cv2.imread('/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/ImagesWithFilters/Individual/Cam1/0.jpg')
image3 = cv2.imread('/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/ImagesWithFilters/Individual/Cam2/0.jpg')
image4 = cv2.imread('/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/ImagesWithFilters/Individual/Cam3/0.jpg')

################################################################################################################################


# # Just in case if we need conversion of colour format
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
# image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)



### If you wnat to apply colour for each image then uncomment this #######
# selected colour = blue
'''image2[:,:,1] = 0 
image2[:,:,2] = 0 
# selected colour = green
image3[:,:,0] = 0 
image3[:,:,2] = 0 
# selected colour = red
image4[:,:,0] = 0 
image4[:,:,1] = 0 '''
###########################################################################



##@################## SET THE REQUIRED RESOLUTION HERE!!! ######################

target_height, target_width = 710, 710

#################################################################################


images = [image1, image2, image3, image4]

# Match resolution 
images = match_resolution(images, target_height, target_width, method="crop")

# Ignore
# print("Image0 shape: ", image1.shape)
# print("Image1 shape: ", image2.shape)
# print("Image2 shape: ", image3.shape)
# print("Image3 shape: ", image4.shape)

# Reference image (Base image for registring images)
reference_image = images[0]


################### ADUST SHIFTS FOR EACH IMAGES HERE ####################

shifts = {
    'image1': {'x': 0, 'y': 5},
    'image2': {'x': -19, 'y': 33},  
    'image3': {'x': -31, 'y': 23},
    'image4': {'x': -22, 'y': 10}
}

##########################################################################


# Translating Images
translated_image1 = translate_image(images[0], shifts['image1']['x'], shifts['image1']['y'])
translated_image2 = translate_image(images[1], shifts['image2']['x'], shifts['image2']['y'])
translated_image3 = translate_image(images[2], shifts['image3']['x'], shifts['image3']['y'])
translated_image4 = translate_image(images[3], shifts['image4']['x'], shifts['image4']['y'])



# Registring images here!!!
#combined_image = np.zeros((target_height, target_width, 3), dtype=np.float32)

reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
translated_image1 = cv2.cvtColor(translated_image1, cv2.COLOR_RGB2GRAY)
translated_image2 = cv2.cvtColor(translated_image2, cv2.COLOR_RGB2GRAY)
translated_image3 = cv2.cvtColor(translated_image3, cv2.COLOR_RGB2GRAY)
translated_image4 = cv2.cvtColor(translated_image4, cv2.COLOR_RGB2GRAY)
reference_image = np.expand_dims(reference_image, axis=-1)
translated_image1 = np.expand_dims(translated_image1, axis=-1)
translated_image2 = np.expand_dims(translated_image2, axis=-1)
translated_image3 = np.expand_dims(translated_image3, axis=-1)
translated_image4 = np.expand_dims(translated_image4, axis=-1)
translated_images = [reference_image, translated_image2, translated_image3, translated_image4]
combined_image = np.concatenate((translated_image1,translated_image2,translated_image3, translated_image4),axis=-1)
combined_image = combined_image.astype(np.uint8)
cv2.imwrite(f'/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/Camera_callibration/callibrated_images/RegisteredImageNew.png', combined_image)

print(combined_image.shape)
print(combined_image)
plt.figure(figsize=(6, 6))
plt.imshow(combined_image.astype(np.uint8))
plt.title('Combined Image')
plt.axis('off')
plt.show()

root = '/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/Camera_callibration/callibrated_images'
image_new = cv2.imread(f'{root}/RegisteredImage.png', cv2.IMREAD_UNCHANGED)
print(image_new.shape)
# # Display all 4 images
# plt.subplot(2,2,1)
# plt.imshow(image1)

# plt.subplot(2,2,2)
# plt.imshow(image2)

# plt.subplot(2,2,3)
# plt.imshow(image3)

# plt.subplot(2,2,4)
# plt.imshow(image4)

# plt.show()



# Here the variable "no_of_imgs" is used to match two imges at a time 
# First we match 2 images (base image and the second image) my changing shifts in x and y, once that is done then we set the value of "no_of_imgs" to 3 and match third image with the base image. We keep on doing this until all the images are completely fused/registered.
'''combined_image = np.array([])
k=0
no_of_imgs = 1               # Select no of images 
for img in translated_images:
    if(k >= no_of_imgs):                 
        break
    k += 1
    #combined_image += img / no_of_imgs
    #combined_image = np.concatenate((reference_image,translated_image2,translated_image3,translated_image4),axis=-1)
    reference_image = np.concatenate((reference_image,img))
    
    #combined_image = fusion(reference_image,translated_image2,translated_image3,translated_image4)
    

# Convert combined image to uint8 for display
#combined_image = combined_image.astype(np.uint8)
print(reference_image.shape)
print(reference_image)
#cv2.imwrite(f'/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/Camera_callibration/callibrated_images/RegisteredImage.png', combined_image)
# Display the combined image
plt.figure(figsize=(6, 6))
plt.imshow(reference_image)
plt.title('Combined Image')
plt.axis('off')
plt.show()
'''