# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:48:51 2023

@author: BigData
"""

import cv2 as cv
import numpy as np

#%%
img1=cv.imread('mot_color70.jpg')[190:350, 440:560]
#img1=cv.imread('soccer_ball.JPG')
gray1=cv.cvtColor(img1, cv.COLOR_BGR2GRAY)


img2=cv.imread('mot_color83.jpg')
#img2=cv.imread('soccer_1.JPG')
gray2=cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

cv.imshow("IMG1", img1);cv.imshow("IMG2", img2)
cv.waitKey()
cv.destroyAllWindows()
#%%
sift=cv.SIFT_create()
kp1, des1 =sift.detectAndCompute(gray1, None)
kp2, des2 =sift.detectAndCompute(gray2, None)
print("특징점 개수", len(kp1), len(kp2))

flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match=flann_matcher.knnMatch(des1, des2, k=2)

T=0.7
good_match=[]
for nearest1, nearest2 in knn_match:
    if(nearest1.distance/nearest2.distance) < T:
       good_match.append(nearest1)
#%%
print(len(good_match)); print(type(good_match))
print(good_match[0].queryIdx, good_match[0].trainIdx,good_match[0].distance, good_match[0].imgIdx)
#%%
#queryIdx: 첫 번째 영상에서 추출한 특징점의 번호, trainIdx: 두 번째 영상에서 추출한 특징점 번호
#distance: 매칭된 두 기술자의 거리, imgIdx: 이미지의 인덱스(단일 영상간의 매칭이므로 0)
points1=np.float32([kp1[gm.queryIdx].pt for gm in good_match]) #매칭 쌍 각각에 대해 특징점 좌표 저장
points2=np.float32([kp2[gm.trainIdx].pt for gm in good_match])

#RANSAC을 사용하여 강인한 호모그래피 행렬 추정
H, mask=cv.findHomography(points1, points2, cv.RANSAC) #H: 추정된 호모그래피 행렬, #mask: 매칭된 점들 중에 인라이어인 점을 1로 표현, 나머지 0
#print(mask) 

h1, w1 = img1.shape[0], img1.shape[1] #첫 번째 영상의 크기
h2, w2 = img2.shape[0], img2.shape[1] #두 번째 영상의 크기

box1=np.float32([[0,0], [0, h1-1], [w1-1, h1-1], [w1-1,0]]).reshape(4,1,2) 
box2=cv.perspectiveTransform(box1,H)
#center 좌표 구하기
print(box2)
print(np.mean(box2[:,0,1]))

img2=cv.polylines(img2, [np.int32(box2)], True, (0,255,0),1)

img_match=np.empty((max(h1, h2), w1+w2, 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow("matches and Homography", img_match)
cv.waitKey()
cv.destroyAllWindows()


