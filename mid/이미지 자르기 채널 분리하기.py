import cv2 as cv
import sys

#%% 이미지 읽기
img = cv.imread('soccer.jpg')  # 이미지를 파일에서 읽어옵니다.
if img is None:
    sys.exit(' No file')  # 이미지가 없으면 프로그램을 종료합니다.

# 원본 이미지 표시
cv.imshow('Original', img)
cv.waitKey()  # 키 입력 대기

# 0부터 1/2까지의 영역을 잘라서 표시
cv.imshow('U half', img[0:img.shape[0]//2, 0:img.shape[1]//2, :])  # 상단 왼쪽 1/4 영역

# 1/4부터 3/4까지의 중앙 영역을 잘라서 표시
cv.imshow('C half', img[img.shape[0]//4:3*img.shape[0]//4, \
                        img.shape[1]//4:3*img.shape[1]//4, :])  # 중앙 영역

# 각 색상 채널 분리하여 표시
cv.imshow('R', img[:, :, 2])  # R 채널
cv.imshow('G', img[:, :, 1])  # G 채널
cv.imshow('B', img[:, :, 0])  # B 채널

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기