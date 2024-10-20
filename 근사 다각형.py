import skimage
import cv2 as cv
import numpy as np
import sys

# skimage에서 샘플 이미지(말)를 로드합니다.
org = skimage.data.horse()

# 이미지가 제대로 로드되었는지 확인합니다.
if org is None:
    sys.exit('No file')  # 이미지가 없으면 프로그램 종료

# 이미지의 형태, 타입, 픽셀 값의 데이터 타입을 출력합니다.
print(org.shape)  # 이미지의 차원 출력
print(type(org))  # 이미지 배열의 타입 출력
print(np.dtype(org[0, 0]))  # 특정 픽셀 값의 데이터 타입 출력

# 픽셀 값을 [0, 255] 범위로 정규화합니다.
print(np.max(np.uint8(org)))  # uint8 형식의 최대 값 출력
print(np.min(np.uint8(org)))  # uint8 형식의 최소 값 출력

# 원본 이미지를 표시합니다.
horse = np.uint8(org)*255
img = 255 - horse
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

cv.imshow("Original",horse)

cv.imshow("Horse", img)  # 반전된 이미지를 표시
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 표시된 창 닫기

# 그레이스케일 이미지를 BGR로 변환하여 색깔 윤곽선을 표시합니다.
img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# 모든 윤곽선을 마젠타 색으로 그립니다.
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow("Contours", img2)  # 윤곽선이 그려진 이미지를 표시
cv.waitKey()  # 키 입력 대기

# 첫 번째 윤곽선의 모멘트를 계산합니다.
m = cv.moments(contours[0])  # 모멘트 정보가 담긴 딕셔너리
print(m)

# 윤곽선의 면적을 계산합니다.
# m00: 영역의 총 픽셀 수 (면적).
# m10: x 좌표의 첫 번째 모멘트 (x 방향의 무게 중심을 계산하는 데 사용).
# m01: y 좌표의 첫 번째 모멘트 (y 방향의 무게 중심을 계산하는 데 사용).
area = cv.contourArea(contours[0])
# 모멘트를 사용하여 중심점을 계산합니다.
cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
# 윤곽선의 둘레를 계산합니다.
perimeter = cv.arcLength(contours[0], True)
# 둥근 정도를 계산합니다.
roundness = 4.0 * np.pi * area / (perimeter * perimeter)
print('면적= ', area, '/n중점=(', cx, cy, ')', '/n둘레= ', perimeter, '/n둥근정도= ', roundness)

# 색상 이미지로 변환하여 추가적인 윤곽선 처리
img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# 윤곽선의 근사 다각형을 계산합니다.
contour_approxP = cv.approxPolyDP(contours[0], 8, True)
print(type(contour_approxP))  # 근사 다각형 타입 출력
print(type([contour_approxP]))

# 근사 다각형을 녹색으로 그립니다.
cv.drawContours(img3, [contour_approxP], -1, (0, 255, 0), 2)

# 윤곽선의 볼록 껍질을 계산합니다.
hull = cv.convexHull(contours[0])
print(hull.shape)  # 볼록 껍질의 형태 출력
print(hull)  # 볼록 껍질의 좌표 출력
print(type(hull))  # 볼록 껍질 타입 출력
hull = hull.reshape(1, hull.shape[0], hull.shape[2])  # 형상 변경
cv.drawContours(img3, hull, -1, (0, 0, 255), 2)  # 볼록 껍질을 빨간색으로 그립니다.

# 윤곽선의 경계 사각형을 계산합니다.
rect = cv.boundingRect(contours[0])
print(rect)  # 경계 사각형 정보 출력
print(type(rect))  # 타입 출력
cv.rectangle(img3, rect, (255, 0, 0), 2)  # 경계 사각형을 파란색으로 그립니다.

# 윤곽선을 둘러싸는 원을 계산합니다.
center, radius = cv.minEnclosingCircle(contours[0])
print('center= ', center, '\nRadius= ', radius)  # 중심과 반지름 출력
cv.circle(img3, np.uint8(center), np.uint8(radius), (0, 255, 255), 2)  # 원을 노란색으로 그립니다.

# 최종 이미지를 표시합니다.
cv.imshow("Horse with Contours", img3)
cv.waitKey()  # 키 입력 대기

cv.destroyAllWindows()  # 모든 창 닫기