import cv2 as cv
import sys

#%% 이미지 읽기
img = cv.imread("rose.png")  # 이미지를 파일에서 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램 종료

#%% 이미지 패치 선택
patch = img[250:350, 170:270, :]  # 특정 영역(패치)을 선택합니다.
# 선택된 패치는 (250, 350)에서 (170, 270)까지의 부분입니다.

#%% 선택한 영역에 사각형 그리기
img = cv.rectangle(img, (170, 250), (270, 350), (255, 0, 0), 3)  
# 선택된 패치의 위치에 빨간색 사각형을 그립니다. 두께는 3픽셀입니다.

#%% 패치 크기 조정
# 최근접 이웃 보간법으로 크기 조정
patch1 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)   
# 양방향 선형 보간법으로 크기 조정
patch2 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)   
# 3차 보간법으로 크기 조정
patch3 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)   

#%% 이미지 출력
cv.imshow('Original', img)  # 원본 이미지 출력
cv.waitKey()  # 키 입력 대기

cv.imshow('Resize Nearest', patch1)  # 최근접 이웃 보간법으로 조정한 이미지 출력
cv.imshow('Resize Linear', patch2)    # 선형 보간법으로 조정한 이미지 출력
cv.imshow('Resize Cubic', patch3)     # 3차 보간법으로 조정한 이미지 출력

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 윈도우 닫기