import cv2
import numpy as np

root = '/Users/gokulmnambiar/Desktop/GitHubRepos/Design-and-Development-of-Multispectral-Camera-for-Aerial-Vehicles'

for i in range(14):

    img = cv2.imread(f'{root}/CalibImgs/calib{i+1}.jpg')

    print(f'img {i} : {img.shape}')
    
    img0 = img[:,240:1040,:]
    img1 = img[:,1520:2320,:]
    img2 = img[:,2800:3600,:]
    img3 = img[:,4080:4880,:]
    
    M1 = cv2.getRotationMatrix2D(((img2.shape[1]-1)/2.0,(img2.shape[0]-1)/2.0),90,1)
    img2 = cv2.warpAffine(img2,M1,(img2.shape[1],img2.shape[0]))
    M2 = cv2.getRotationMatrix2D(((img3.shape[1]-1)/2.0,(img3.shape[0]-1)/2.0),180,1)
    img3 = cv2.warpAffine(img3,M2,(img3.shape[1],img3.shape[0]))

    cv2.imwrite(f'{root}/NewCalibImgs/Cam0/{i+1}.jpg', img0)
    cv2.imwrite(f'{root}/NewCalibImgs/Cam1/{i+1}.jpg', img1)
    cv2.imwrite(f'{root}/NewCalibImgs/Cam2/{i+1}.jpg', img2)
    cv2.imwrite(f'{root}/NewCalibImgs/Cam3/{i+1}.jpg', img3)
#cv2.imshow('img0',img0)
#cv2.waitKey(0)
#cv2.destroyAllWindows()