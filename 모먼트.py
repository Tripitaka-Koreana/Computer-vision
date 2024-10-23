import skimage
import cv2 as cv
import numpy as np
import sys
import math

#%% 이미지 불러오기 및 처리

# 스키이미지에서 'horse' 데이터셋 로드
org = skimage.data.horse()

# 이미지가 로드되지 않았을 경우 프로그램 종료
if org is None:
    sys.exit('No file')

# 이미지의 정보 출력
print(org.shape)  # 이미지의 크기 출력
print(type(org))  # 이미지 배열의 타입 출력
print(np.dtype(org[0, 0]))  # 첫 번째 픽셀의 데이터 타입 출력
print(np.max(np.uint8(org)))  # 이미지의 최대값 출력
print(np.min(np.uint8(org)))  # 이미지의 최소값 출력

# 원본 이미지 출력
cv.imshow("Original", np.uint8(org) * 255)  # 이미지를 255로 스케일링하여 출력
img = 255 - np.uint8(org) * 255  # 흑백 반전 처리
cv.imshow("Horse", img)  # 반전된 이미지 출력
cv.waitKey()
cv.destroyAllWindows()

#%% 윤곽선 찾기

# 윤곽선 찾기 (외부 윤곽선)
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# 윤곽선을 그리기 위한 이미지 생성
img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)  # 윤곽선 그리기
cv.imshow("Contours", img2)  # 윤곽선이 그려진 이미지 출력
cv.waitKey()

#%% 윤곽선의 모멘트 계산

m = cv.moments(contours[0])  # 첫 번째 윤곽선의 모멘트 계산 (사전 형태로 반환)
print(m)  # 모멘트 출력
area = cv.contourArea(contours[0])  # 윤곽선의 면적 계산
cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']  # 중심 좌표 계산
perimeter = cv.arcLength(contours[0], True)  # 윤곽선의 둘레 길이 계산
roundness = 4.0 * np.pi * area / (perimeter * perimeter)  # 둥근 정도 계산
print('면적= ', area, '/n중점=(', cx, cy, ')', '/n둘레= ', perimeter, '/n둥근정도= ', roundness)

#%% 윤곽선 근사 및 기타 기하학적 특성

# 윤곽선 근사
img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # 이미지 색상 변환
contour_approxP = cv.approxPolyDP(contours[0], 8, True)  # 윤곽선 근사
print(type(contour_approxP))  # 근사된 윤곽선의 타입 출력
print(type([contour_approxP]))  # 리스트 형태로 변환한 근사 윤곽선 타입 출력

cv.drawContours(img3, [contour_approxP], -1, (0, 255, 0), 2)  # 근사 윤곽선 그리기

# 볼록 껍질 계산
hull = cv.convexHull(contours[0])  # 윤곽선의 볼록 껍질 계산
print(hull.shape)  # 볼록 껍질의 크기 출력
print(hull)  # 볼록 껍질 좌표 출력
print(type(hull))  # 볼록 껍질의 타입 출력 

# 볼록 껍질을 이미지에 그리기 위해 형상 변경
hull = hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours(img3, hull, -1, (0, 0, 255), 2)  # 볼록 껍질 그리기

# 윤곽선의 경계 사각형 계산
rect = cv.boundingRect(contours[0])  # 경계 사각형 계산
print(rect)  # 경계 사각형 좌표 출력
print(type(rect))  # 경계 사각형의 타입 출력 
cv.rectangle(img3, rect, (255, 0, 0), 2)  # 경계 사각형 그리기

# 최소 둘레 원 계산
center, radius = cv.minEnclosingCircle(contours[0])  # 최소 둘레 원의 중심과 반지름 계산
print('center= ', center, '\nRadius= ', radius)  # 중심과 반지름 출력
cv.circle(img3, np.uint8(center), np.uint8(radius), (0, 255, 255), 2)  # 최소 둘레 원 그리기

# 최종 결과 출력
cv.imshow("Horse with Contours", img3)  # 윤곽선, 볼록 껍질, 경계 사각형, 최소 둘레 원이 그려진 이미지 출력
cv.waitKey()
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 종료
