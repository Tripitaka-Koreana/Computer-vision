import cv2 as cv
import sys
import numpy as np

#%% 이미지 읽기
img = cv.imread('soccer.jpg')  # 이미지를 파일에서 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램을 종료합니다.

#%% R 채널에 대해 이진화
# Otsu의 이진화 방법을 사용하여 R 채널을 이진화합니다.
# cv.THRESH_BINARY | cv.THRESH_OTSU를 사용하여 이진화 임계값을 자동으로 계산합니다.
random_matrix = np.array([[123, 124, 123, 124, 125],
                           [135, 145, 135, 145, 155],
                           [128, 130, 125, 128, 132],
                           [120, 122, 121, 123, 125],
                           [130, 135, 130, 135, 140]], dtype=np.uint8) # 임의배열

t, bin = cv.threshold(random_matrix, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# 계산된 임계값 출력
print("threshold", t)  # 사용된 임계값 출력

# R 채널과 이진화된 결과를 각각 표시
cv.imshow('R', img[:,:,2])  # 원본 R 채널 이미지 표시
cv.imshow('R binary', bin)    # 이진화된 R 채널 이미지 표시

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기