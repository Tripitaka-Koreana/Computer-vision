import cv2 as cv
import numpy as np
import sys
import time

#%% 사용자 정의 그레이스케일 변환 함수 1
def my_cvtGray1(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])  # 그레이스케일 이미지 초기화
    # 이미지의 모든 픽셀에 대해 루프 실행
    for r in range(bar_img.shape[0]):
        for c in range(bar_img.shape[1]):
            # 각 픽셀의 RGB 값을 가중치로 조합하여 그레이스케일로 변환
            g[r, c] = 0.114 * bar_img[r, c, 0] + 0.587 * bar_img[r, c, 1] + 0.299 * bar_img[r, c, 2]
    return np.uint8(g)  # uint8 타입으로 변환하여 반환

#%% 사용자 정의 그레이스케일 변환 함수 2
def my_cvtGray2(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])  # 그레이스케일 이미지 초기화
    # 배열 슬라이싱을 사용하여 그레이스케일로 변환
    g = 0.114 * bar_img[:, :, 0] + 0.587 * bar_img[:, :, 1] + 0.299 * bar_img[:, :, 2]
    return np.uint8(g)  # uint8 타입으로 변환하여 반환

#%% 이미지 읽기
img = cv.imread("girl_laughing.jpg")  # 이미지를 파일에서 읽어옵니다.
if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램 종료

#%% 그레이스케일 변환 시간 측정
start = time.time()  # 시작 시간 기록
my_cvtGray1(img)  # 사용자 정의 함수 1 호출
print('My time1: ', time.time() - start)  # 소요 시간 출력

start = time.time()  # 시작 시간 기록
my_cvtGray2(img)  # 사용자 정의 함수 2 호출
print('My time2: ', time.time() - start)  # 소요 시간 출력

start = time.time()  # 시작 시간 기록
cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # OpenCV의 그레이스케일 변환 함수 호출
print('MOpenCV time1: ', time.time() - start)  # 소요 시간 출력