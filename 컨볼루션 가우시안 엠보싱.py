import cv2 as cv
import numpy as np
import sys

#%% 이미지 읽기
img = cv.imread("soccer.jpg")  # 이미지를 파일에서 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램 종료

#%% 배열 생성 및 출력
a = np.array([-3, -2, -1, 0, 254, 255, 256, 257, 258], dtype=np.uint8)
print(type(a))  # 배열의 타입 출력
print(type(a[0]))  # 배열 첫 번째 요소의 타입 출력
print(a)  # 배열 출력
# 주석: 결과는 [253 254 255   0 254 255   0   1   2]로, uint8의 특성으로 인해
# 256 이상의 값은 0~255 범위로 변환됩니다.

#%% 이미지 타입 확인
print(type(img))  # 이미지 배열의 타입 출력
print(type(img[0, 0, 0]))  # 첫 번째 픽셀의 첫 번째 채널 타입 출력

#%% 이미지 크기 조정 및 그레이스케일 변환
img = cv.resize(img, dsize=(0, 0), fx=0.4, fy=0.4)  # 이미지 크기를 40%로 조정
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환

# 이미지에 텍스트 추가
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  
# 텍스트 위치, 크기 및 색상 지정
cv.imshow('Original', gray)  # 원본 이미지 출력
cv.waitKey()  # 키 입력 대기

#%% 이미지 블러링
smooth = np.hstack((
    cv.GaussianBlur(gray, (5, 5), 0.0), 
    cv.GaussianBlur(gray, (9, 9), 0.0), 
    cv.GaussianBlur(gray, (15, 15), 0.0)
))  # 다양한 커널 크기로 가우시안 블러 적용 후 수평으로 결합

# 주석: hstack은 수평으로 이미지를 결합하는 함수입니다.
# 커널 크기가 커질수록 이미지가 더 흐려집니다.
cv.imshow('Smooth', smooth)  # 블러 처리된 이미지 출력
cv.waitKey()  # 키 입력 대기

#%% 엠보싱 필터 적용
femboss = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 엠보싱 필터 정의

# 이미지 타입을 uint16으로 변경하여 큰 값 처리
gray16 = np.uint16(gray)

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 윈도우 닫기

# 엠보싱 처리: 255를 넘어가는 값을 클리핑하여 원래 범위로 조정
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255)) #-1인자: 입력영상과 동일한 dtype
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss) + 128)
emboss_worse = cv.filter2D(gray, -1, femboss)  # 엠보싱 적용

# 각 엠보싱 이미지 출력
cv.imshow('Emboss', emboss)  # 정상 엠보싱
cv.imshow('Emboss_bad', emboss_bad)  # 잘못된 엠보싱
cv.imshow('Emboss_worse', emboss_worse)  # 원본 이미지에 엠보싱 적용

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 윈도우 닫기