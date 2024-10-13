import cv2 as cv
import sys
import numpy as np

#%% 이미지 파일 읽기
img = cv.imread('BALLOON.BMP')  # 이미지 파일 읽기
if img is None:
    sys.exit("No File exists.")  # 파일이 존재하지 않을 경우 프로그램 종료

# 원본 이미지 표시
cv.imshow("Original", img)  # 원본 이미지 보여주기
cv.waitKey()  # 키 입력 대기

#%% BGR에서 HSV로 변환
# cv.cvtColor(img, cv.COLOR_BGR2HSV, dst=dst_hsv) 랑 dst_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 같음
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # BGR 색상 공간을 HSV 색상 공간으로 변환

# 색상 범위에 따른 마스크 생성
mask = cv.inRange(hsv, (91, 0, 80), (125, 255, 255))  # 지정된 색상 범위에 해당하는 부분을 흰색으로, 나머지는 검은색으로 설정
cv.imshow("Mask", mask)  # 생성된 마스크 이미지 보여주기
cv.waitKey()  # 키 입력 대기

#%% 결과 이미지 생성
dst = np.full(img.shape, 255, dtype=np.uint8)  # 흰색 배경으로 채운 배열 생성

# 마스크를 사용하여 원본 이미지에서 검출된 부분만 가져오기
detected = cv.copyTo(img, mask, dst)  # 마스크를 적용하여 원본 이미지에서 검출된 부분만 dst에 복사
cv.imshow("Result", detected)  # 최종 결과 이미지 보여주기
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기