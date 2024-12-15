import cv2 as cv
import matplotlib.pyplot as plt
import sys

#%% 이미지 읽기
img = cv.imread("mistyroad.jpg")  # 이미지를 파일에서 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램 종료

print(img.shape)  # 이미지의 차원 정보 출력

#%% 그레이스케일 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환
print(gray.shape)  # 그레이스케일 이미지의 차원 정보 출력

#plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([])  # 그레이스케일 이미지 출력 (주석 처리됨)

#%% 히스토그램 계산 및 출력
h = cv.calcHist([gray], [0], None, [256], [0, 255])  # 그레이스케일 이미지의 히스토그램 계산
plt.plot(h, color='r', linewidth=1)  # 히스토그램을 빨간색으로 출력
plt.show()  # 그래프 표시

#%% 히스토그램 균등화
equal = cv.equalizeHist(gray)  # 그레이스케일 이미지에 히스토그램 균등화 적용
#plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([])  # 균등화된 이미지 출력 (주석 처리됨)

#%% 균등화 후 히스토그램 계산 및 출력
h = cv.calcHist([equal], [0], None, [256], [0, 255])  # 균등화된 이미지의 히스토그램 계산
plt.plot(h, color='r', linewidth=1)  # 히스토그램을 빨간색으로 출력
plt.show()  # 그래프 표시