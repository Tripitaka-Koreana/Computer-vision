import cv2 as cv  # 이미지 처리를 위한 OpenCV 라이브러리 임포트
import sys  # 시스템 관련 매개변수 및 함수 사용을 위한 sys 임포트
import numpy as np  # 수치 계산을 위한 NumPy 임포트

# 이미지 파일 읽기
img = cv.imread('soccer.jpg')
if img is None:
    sys.exit("파일이 존재하지 않습니다.")  # 이미지 파일이 없으면 종료

# 원본 이미지 표시
cv.imshow("Original", img)
cv.waitKey()  # 키 입력 대기

# 이미지를 BGR에서 HSV 색상 공간으로 변환
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('Hsv', hsv)  # HSV 이미지 표시
cv.waitKey()  # 키 입력 대기

# HSV 이미지를 세 개의 채널(Hue, Saturation, Value)로 분리
h, s, v = cv.split(hsv)

# Hue 채널 표시
cv.imshow('Hue', h)
cv.waitKey()  # 키 입력 대기

# 디버깅용으로 Hue 채널 값 출력 (주석 해제 시)
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(h)

# Hue 채널에서 최대 값 출력
print(np.max(h))

# Saturation 채널 표시
cv.imshow('Sat', s)
cv.waitKey()  # 키 입력 대기

# Value 채널 표시
cv.imshow('Value', v)
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()

#채널 별 작업 후 병합
#명도 채널 값을 255로 변환 후 병합
print(v.ndim)
print(v.shape)
print(v)
v=np.zeros((948, 1434), np.uint8)+255
dst=cv.merge([h,s,v])
new=cv.cvtColor(dst,cv.COLOR_HSV2BGR)
cv.imshow('New', new)
cv.waitKey()
cv.destroyAllWindows()
