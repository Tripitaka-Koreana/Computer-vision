import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv.imread('soccer.jpg')  # 이미지를 파일에서 읽어옵니다.

# R 채널의 히스토그램 계산
h = cv.calcHist([img], [2], None, [256], [0, 256])  # R 채널 (인덱스 2)의 히스토그램을 계산합니다.

# 히스토그램 출력
print(h)  # 계산된 히스토그램 값 출력
print(len(h))  # 히스토그램의 길이 출력 (256)
print(type(h))  # 히스토그램의 데이터 타입 출력

# 히스토그램 플롯
plt.plot(h, color='r', linewidth=1)  # R 채널의 히스토그램을 빨간색으로 플롯합니다.
plt.title('Histogram of Red Channel')  # 그래프 제목 설정
plt.xlabel('Pixel Intensity')  # x축 레이블 설정
plt.ylabel('Frequency')  # y축 레이블 설정
plt.show()  # 그래프 표시