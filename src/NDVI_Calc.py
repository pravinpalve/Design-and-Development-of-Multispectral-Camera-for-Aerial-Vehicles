import numpy as np
import cv2
import matplotlib.pyplot as plt

root = '/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/Camera_callibration/callibrated_images'

image = cv2.imread(f'{root}/RegisteredImageOld.png', cv2.IMREAD_UNCHANGED)

print(image.shape)

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title('Image')
plt.axis('off')
plt.show()

NDVI = (image[:,:,3].astype(np.float32)-image[:,:,1].astype(np.float32))/((image[:,:,3].astype(np.float32)+image[:,:,1].astype(np.float32)))
NDVI = np.where((image[:,:,3].astype(np.float32)+image[:,:,1].astype(np.float32)) == 0, 0, NDVI)

'''NDVI = (image[:,:,3]-image[:,:,1])/((image[:,:,3]+image[:,:,1]))
NDVI = np.where((image[:,:,3]+image[:,:,1]) == 0, 0, NDVI)'''


print(image[:,:,2])

print(NDVI.shape)

result = cv2.normalize(NDVI, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imwrite('/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/Result/NDVI/imgOld.png', result)

plt.figure(figsize=(6, 6))
plt.imshow(NDVI)
plt.title('NDVI')
plt.axis('off')
plt.show()
