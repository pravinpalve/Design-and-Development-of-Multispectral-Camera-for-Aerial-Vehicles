import cv2
import numpy as np

root = '/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles/CalibImgs'

for i in range(14):

    img = cv2.imread(f'{root}/calib{i+1}.jpg')

    print(img.shape)

    img0 = img[:,0:1280,:]
    img1 = img[:,1280:2560,:]
    img2 = img[:,2560:3840,:]
    img3 = img[:,3840:5120,:]

    cv2.imwrite(f'{root}/NewCalibImgs/Cam0/{i+1}.jpg', img0)
    cv2.imwrite(f'{root}/NewCalibImgs/Cam1/{i+1}.jpg', img1)
    cv2.imwrite(f'{root}/NewCalibImgs/Cam2/{i+1}.jpg', img2)
    cv2.imwrite(f'{root}/NewCalibImgs/Cam3/{i+1}.jpg', img3)
#cv2.imshow('img0',img0)
#cv2.waitKey(0)
#cv2.destroyAllWindows()