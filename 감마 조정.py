import cv2 as cv
import numpy as np
import sys

#%% 이미지 읽기
img = cv.imread("soccer.jpg")  # 이미지를 파일에서 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램을 종료합니다.

# 이미지 크기 조정 (25%로 축소)
img = cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)

# 감마 조정 함수 정의
def gamma(f, gamma=1.0):
    f1 = f / 255.0  # 픽셀 값을 [0, 1] 범위로 정규화
    return np.uint8(255 * (f1 ** gamma))  # 감마 조정을 적용한 후 다시 [0, 255] 범위로 변환

# 다양한 감마 값을 적용하여 이미지를 나란히 배치
gc = np.hstack((
    gamma(img, 0.5),  # 감마 0.5
    gamma(img, 0.75), # 감마 0.75
    gamma(img, 1.0),  # 감마 1.0 (원본 이미지)
    gamma(img, 2.0),  # 감마 2.0
    gamma(img, 3.0)   # 감마 3.0
))    

# 감마 조정 결과를 표시
cv.imshow("gamma", gc)  # 감마 조정된 이미지를 나란히 표시

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기