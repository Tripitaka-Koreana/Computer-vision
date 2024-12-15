import cv2 as cv
import sys
import numpy as np

#%% 이미지 읽기
img = cv.imread('boldt.jpg')  # 이미지를 파일에서 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램을 종료합니다.

#%% R 채널에 대해 이진화
# Otsu의 이진화 방법을 사용하여 R 채널을 이진화합니다.
# cv.THRESH_BINARY | cv.THRESH_OTSU를 사용하여 이진화 임계값을 자동으로 계산합니다.
"""random_matrix = np.array([[123, 124, 123, 124, 125],
                           [135, 145, 135, 145, 155],
                           [128, 130, 125, 128, 132],
                           [120, 122, 121, 123, 125],
                           [130, 135, 130, 135, 140]], dtype=np.uint8) # 임의배열
"""

#R채널 오츄알고리즘을 사용한 임계값 구하기
#threshold(이미지, 임계값(0), MaxValue, 이진화 방식) -> 임계값, 이진화된 영상
#t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 동일
t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 계산된 임계값 출력
print("threshold", t)  # 사용된 임계값 출력

# R 채널과 이진화된 결과를 각각 표시
cv.imshow('R', img[:,:,2])  # 원본 R 채널 이미지 표시
cv.imshow('R binary', bin)    # 이진화된 R 채널 이미지 표시

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

"""
cv2.THRESH_BINARY: 임계값 이하 = 0, 임계값 초과 = maxVal.
cv2. THRESH_BINARY_INV: 임계값 이하 = maxVal, 임계값 초과 = 0.
cv2. THRESH_TRUNC: 임계값 이하 = 그대로, 임계값 초과 = threshold.
cv2. THRESH_TOZERO: 임계값 이하 = 0, 임계값 초과 = src(x,y).
cv2. THRESH_TOZERO_INV: 임계값 이하 = src(x,y), 임계값 초과 = 0.
"""

