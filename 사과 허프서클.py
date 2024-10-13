import cv2 as cv
import numpy as np

#%% 
# 이미지를 불러옵니다.
img = cv.imread('apples.jpg')
cv.imshow('Apple', img)  # 원본 이미지를 화면에 표시합니다.

cv.waitKey()  # 키 입력을 기다립니다.

#%% 
# 이미지를 그레이스케일로 변환합니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 허프 변환을 사용하여 원을 찾습니다.
apples = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=50, maxRadius=120)

# 찾은 원의 정보 출력
print(apples.shape)  # 찾은 원의 개수 및 정보의 형태를 출력합니다.
print(apples)  # 찾은 원의 좌표 및 반지름 정보를 출력합니다.

#%% 
# 찾은 원을 원본 이미지에 그립니다.
for i in apples[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)  # 원의 중심과 반지름을 사용해 원을 그림

cv.imshow('Apple detection', img)  # 원이 그려진 이미지를 화면에 표시합니다.

cv.waitKey()  # 키 입력을 기다립니다.
cv.destroyAllWindows()  # 모든 창을 닫습니다.