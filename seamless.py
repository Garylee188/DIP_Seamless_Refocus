'''
Reading Files
'''
import glob
import cv2

files = glob.glob('./ImagesJPG/ImagesJPG/*jpg')
files.sort() # So focal lenghth value from small to large
print(files)

focal_stack = []
for file in files:
  img = cv2.imread(file)
  focal_stack.append(img)
# print(len(focal_stack))

#remove checkerboard
# focal_stack = focal_stack[:-7]
# print(len(focal_stack))

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

def SIFT_seamless(img_base, img_test):
    ### Use SIFT to find keypoints between test and base ###
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_test, None)
    kp2, des2 = sift.detectAndCompute(img_base, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio = 0.5
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])

    ### Match test image to base image, to make it seamless ###
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
  
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
    
    # print(M)
    return M

img_base_0 = cv2.imread('./ImagesJPG/ImagesJPG/IMAG2121.jpg', cv2.IMREAD_GRAYSCALE)  # the nearest image
img_test = cv2.imread('./ImagesJPG/ImagesJPG/IMAG2131.jpg', cv2.IMREAD_GRAYSCALE)  # test other image to match img_base
img_base_0_sharp = (255 * filters.unsharp_mask(img_base_0, radius=30, amount=1)).astype('uint8')
img_test_sharp = (255 * filters.unsharp_mask(img_test, radius=30, amount=1)).astype('uint8')
Matrix_base = SIFT_seamless(img_base_0_sharp, img_test_sharp)
img_seamless = cv2.warpPerspective(img_test_sharp, Matrix_base, (img_base_0.shape[1], img_base_0.shape[0]))
cv2.imwrite('./test_0.jpg', img_base_0)
cv2.imwrite('./test.jpg', img_seamless)

from tqdm import tqdm
import os

path = "./seamless_Image/"
if os.path.exists(path) == False:
  os.makedirs(path)

img_base = cv2.imread('./ImagesJPG/ImagesJPG/IMAG2131.jpg', cv2.IMREAD_GRAYSCALE)  # the nearest image
for image in tqdm(files):
    name = os.path.basename(image)
    img_test = cv2.imread(f'./ImagesJPG/ImagesJPG/{name}', cv2.IMREAD_GRAYSCALE)  # test other image to match img_base
    img_test_bgr = cv2.imread(f'./ImagesJPG/ImagesJPG/{name}')
    img_base_sharp = (255 * filters.unsharp_mask(img_base, radius=30, amount=1)).astype('uint8')
    img_test_sharp = (255 * filters.unsharp_mask(img_test, radius=30, amount=1)).astype('uint8')
    Matrix = SIFT_seamless(img_base_sharp, img_test_sharp)
    # Matrix_base = Matrix_base * Matrix
    img_seamless = cv2.warpPerspective(img_test_bgr, Matrix, (img_base.shape[1], img_base.shape[0]))
    img_seamless = cv2.warpPerspective(img_seamless, Matrix_base, (img_base_0.shape[1], img_base_0.shape[0]))
    #   img_seamless = cv2.resize(img_seamless,(1440,1080))
    cv2.imwrite(f"./seamless_Image/{name}",img_seamless)

img = cv2.imread('./seamless_Image/0.jpg', cv2.IMREAD_GRAYSCALE)
img2 = img.copy()
ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
# cv2.imwrite('./thresh.png', thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
cv2.drawContours(img, contours, 0, (0,255,0), 2)

areas = []
for c in range(len(contours)):
    areas.append(cv2.contourArea(contours[c]))
max_id = areas.index(max(areas))
max_rect = cv2.minAreaRect(contours[max_id])
# print(max_rect)
max_box = cv2.boxPoints(max_rect)
max_box = np.int0(max_box)
print(max_box)
cv2.drawContours(img2, [max_box], 0, (0, 255, 0), 2)
# cv2.imwrite('./contours_2.png', img2)

seamless_files = glob.glob('./seamless_newImage/*jpg')
seamless_files.sort() # So focal lenghth value from small to large
# print(files)
path = "./seamless_newImage_crop/"
if os.path.exists(path) == False:
  os.makedirs(path)

for image in tqdm(files):
    name = os.path.basename(image)
    img_seamless = cv2.imread(f'./seamless_newImage/{name}')
    dst_img = img_seamless[max_box[0][1]:max_box[2][1], max_box[0][0]:max_box[1][0]]
    cv2.imwrite(f'./seamless_newImage_crop/{name}', dst_img)

