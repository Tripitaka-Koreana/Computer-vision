import cv2 as cv
import numpy as np

#%% 이미지 읽기 및 전처리
img = cv.imread("hand_sample2.jpg")  # 이미지를 읽어옴
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환

# Otsu의 방법을 이용해 이진화 (반전)
#t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
t, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)


# 원본 이미지, 그레이 이미지, 이진 이미지 표시
cv.imshow("Binary", binary)
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 창 닫기

#%% 전경 및 배경 마스크 생성
kernel = cv.getStructuringElement(cv.MORPH_RECT, (8,8))  # 8x8 커널 생성
fg = cv.erode(binary, kernel, iterations=7)  # 이진 이미지에 침식을 적용하여 전경 추출
cv.imshow("Foreground", fg)  # 전경 이미지 표시
cv.waitKey()

bg = cv.dilate(binary, kernel, iterations=7)  # 이진 이미지에 팽창을 적용하여 배경 추출
t, bg = cv.threshold(bg, 1, 128, cv.THRESH_BINARY_INV)  # 배경 이진화
cv.imshow("Background", bg)  # 배경 이미지 표시
cv.waitKey()

#%% 마커 설정
markers = fg + bg  # 전경과 배경을 결합하여 마커 생성
cv.imshow("Markers", markers)  # 마커 이미지 표시
cv.waitKey()

markers = np.int32(markers)  # 마커를 정수형으로 변환
markers = cv.watershed(img, markers)  # 워터셰드 알고리즘 적용


#%% 결과 이미지 생성
dst = np.ones(img.shape, np.uint8) * 255  # 흰색 배경 이미지 생성
dst[markers == -1] = 0  # 경계(워터셰드 경계) 부분을 검은색으로 설정
cv.imshow("Watershed", dst)  # 워터셰드 결과 이미지 표시

# 원본 이미지에 경계 표시
img[markers == -1] = [255, 0, 0]  # 경계를 표시
cv.imshow("Color&Boundary", img)  # 경계가 표시된 원본 이미지 표시
cv.waitKey()

#%% 마커 후처리 및 표시
markers = np.uint8(np.clip(markers, 0, 255))  # 마커를 0-255 범위로 클리핑하고 8비트 형식으로 변환
cv.imshow("Segmentation", markers)  # 세그멘테이션 결과 이미지 표시
cv.waitKey()

cv.destroyAllWindows()  # 모든 창 닫기