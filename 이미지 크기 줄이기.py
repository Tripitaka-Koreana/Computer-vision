import cv2 as cv
import sys

#%% 이미지 파일 읽기
img = cv.imread('soccer.jpg')

# 이미지 파일이 존재하지 않을 경우 프로그램 종료
if img is None:
    sys.exit("No File exists.")

# 이미지의 타입과 형태 출력
print(type(img))  # 이미지 타입 출력
print(img.shape)  # 이미지의 형태 (높이, 너비, 채널 수) 출력

#%% 색상을 회색조로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR 이미지를 회색조로 변환
print(gray.shape)  # 회색조 이미지의 형태 출력

# 회색조 이미지를 크기 조정 (50% 축소)
gray_small = cv.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)  # 크기 축소
print(gray_small.shape)  # 축소된 회색조 이미지의 형태 출력

# 회색조 이미지와 축소된 회색조 이미지를 파일로 저장
cv.imwrite('soccer_gray.jpg', gray)  # 회색조 이미지 저장
cv.imwrite('soccer_gray_small.jpg', gray_small)  # 축소된 회색조 이미지 저장

#%% 이미지 표시
cv.imshow('Color', img)  # 원본 컬러 이미지 표시
cv.imshow('Gray', gray)  # 회색조 이미지 표시
cv.imshow('Gray Resized', gray_small)  # 축소된 회색조 이미지 표시

# 키 입력을 대기하고 모든 창 닫기
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 창 닫기