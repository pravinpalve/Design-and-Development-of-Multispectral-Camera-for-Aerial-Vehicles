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
image1 = cv2.imread(r'')
image2 = cv2.imread(r'')
image3 = cv2.imread(r'')
image4 = cv2.imread(r'')

################################################################################################################################


# # Just in case if we need conversion of colour format
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
# image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)



### If you wnat to apply colour for each image then uncomment this #######
# selected colour = blue
image2[:,:,1] = 0 
image2[:,:,2] = 0 
# selected colour = green
image3[:,:,0] = 0 
image3[:,:,2] = 0 
# selected colour = red
image4[:,:,0] = 0 
image4[:,:,1] = 0 
###########################################################################



##@################## SET THE REQUIRED RESOLUTION HERE!!! ######################

target_height, target_width = 690, 690

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
    'image2': {'x': 8, 'y': 18},  
    'image3': {'x': 22, 'y': 15},
    'image4': {'x': 12, 'y': 6}
}

##########################################################################


# Translating Images
translated_image2 = translate_image(images[1], shifts['image2']['x'], shifts['image2']['y'])
translated_image3 = translate_image(images[2], shifts['image3']['x'], shifts['image3']['y'])
translated_image4 = translate_image(images[3], shifts['image4']['x'], shifts['image4']['y'])



# Registring images here!!!
combined_image = np.zeros((target_height, target_width, 3), dtype=np.float32)
translated_images = [reference_image, translated_image2, translated_image3, translated_image4]


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

k=0
no_of_imgs = 2               # Select no of images 
for img in translated_images:
    if(k >= no_of_imgs):                 
        break
    k += 1
    combined_image += img / no_of_imgs
    

# Convert combined image to uint8 for display
combined_image = combined_image.astype(np.uint8)

# Display the combined image
plt.figure(figsize=(6, 6))
plt.imshow(combined_image)
plt.title('Combined Image')
plt.axis('off')
plt.show()
