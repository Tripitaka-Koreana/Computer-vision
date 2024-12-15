
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

#%% 이미지 읽기
img = cv.imread("JohnHancocksSignature.png", cv.IMREAD_UNCHANGED)
# cv.IMREAD_UNCHANGED: 이미지의 모든 채널(알파 채널 포함)을 그대로 유지합니다.
# 알파 채널을 포함한 4채널 이미지
imgc = cv.imread("JohnHancocksSignature.png")
# 알파 채널을 무시하고 3채널(BGR) 이미지로 읽습니다. 3채널 이미지

# 이미지가 존재하지 않으면 프로그램 종료
if img is None:
    sys.exit("No File exists.")

# 이미지 정보 출력
print(type(img))  # img의 데이터 타입 출력
print(img.shape)  # img의 차원 출력 (채널 수 포함)
print(imgc.shape) # imgc의 차원 출력 (채널 수 포함)
print(img)        # img의 픽셀 값 출력
print(imgc)       # imgc의 픽셀 값 출력
print(np.max(imgc))  # imgc에서 최대값 출력
print(np.max(img[:,:,3]))  # 알파 채널에서 최대값 출력
print(np.min(img[:,:,3]))  # 알파 채널에서 최소값 출력

# 알파 채널을 이미지로 표시
cv.imshow("unchanged", img[:,:,3])  # 알파 채널을 그레이스케일 이미지로 표시
cv.waitKey()  # 키 입력 대기
cv.imshow("color", imgc)  # 색상 이미지 표시
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

#%% 이진화
# Otsu의 방법을 사용하여 알파 채널을 이진화
t, binimg = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("Binary", binimg)  # 이진화된 이미지 표시
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

#%% 이진화된 이미지 시각화
plt.imshow(binimg, cmap='gray')  # 이진화된 이미지를 그레이스케일로 표시
plt.xticks([])  # x축 눈금 제거
plt.yticks([])  # y축 눈금 제거
plt.show()  # 그래프 표시

# 이진화된 이미지의 하반부 및 좌측 절반 추출
b = binimg[binimg.shape[0]//2: binimg.shape[0], 0: binimg.shape[0]//2+1]
plt.imshow(b, cmap='gray')  # 추출된 이미지를 그레이스케일로 표시
plt.xticks([])  # x축 눈금 제거
plt.yticks([])  # y축 눈금 제거
plt.show()  # 그래프 표시

#%% 마스크 정의
# 구조적 요소 정의 (5x5 크기)
se = np.uint8([[0,0,1,0,0],
                [0,1,1,1,0],
                [1,1,1,1,1],
                [0,1,1,1,0],
                [0,0,1,0,0]])
print(se.shape)  # 구조적 요소의 크기 출력

# 팽창 (Dilation)
b_dilation = cv.dilate(b, se, iterations=1)  # 이진 이미지에 팽창 적용
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]); plt.yticks([])  # 팽창된 이미지 표시
plt.show()  # 그래프 표시

# 침식 (Erosion)
b_eroision = cv.erode(b, se, iterations=1)  # 이진 이미지에 침식 적용
plt.imshow(b_eroision, cmap='gray'), plt.xticks([]); plt.yticks([])  # 침식된 이미지 표시
plt.show()  # 그래프 표시

# 닫힘 (Closing)
b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)  # 팽창 후 침식을 적용하여 닫힘 수행
plt.imshow(b_closing, cmap='gray'), plt.xticks([]); plt.yticks([])  # 닫힘 결과 이미지 표시
plt.show()  # 그래프 표시

# 열림 (Opening)
b_opening = cv.dilate(cv.erode(b, se, iterations=1), se, iterations=1)  # 팽창 후 침식을 적용하여 닫힘 수행
plt.imshow(b_opening, cmap='gray'), plt.xticks([]); plt.yticks([])  # 닫힘 결과 이미지 표시
plt.show()  # 그래프 표시