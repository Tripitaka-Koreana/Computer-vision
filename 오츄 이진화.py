import cv2 as cv
import sys
import numpy as np

#%% 이미지 읽기
img = cv.imread('bear.jpg')  # 이미지를 파일에서 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램을 종료합니다.

imgg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
t, bin = cv.threshold(imgg, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# 계산된 임계값 출력
print("threshold", t)  # 사용된 임계값 출력

# R 채널과 이진화된 결과를 각각 표시
cv.imshow('R', img[:,:,2])  # 원본 R 채널 이미지 표시
cv.imshow('R binary', bin)    # 이진화된 R 채널 이미지 표시

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기