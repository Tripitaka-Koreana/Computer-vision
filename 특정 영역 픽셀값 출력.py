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
type(img)

# 픽셀 요소 출력 (첫 번째 픽셀의 B, G, R 값)
print(img[0, 0, 0], img[0, 0, 1], img[0, 0, 2])  # (0,0) 위치의 픽셀 값
print(img[0, 1, 0], img[0, 1, 1], img[0, 1, 2])  # (0,1) 위치의 픽셀 값

# 첫 10x10 영역의 각 채널(B, G, R) 값 출력
print(img[0:10, 0:10, 0], img[0:10, 0:10, 1], img[0:10, 0:10, 2])  

# 이미지 전체와 부분 이미지를 화면에 표시
cv.imshow('Img Display', img)  # 전체 이미지 표시
cv.imshow("Partial", img[0:100, 0:100, :])  # 10x10 부분 이미지 표시

# 키 입력을 대기하고 모든 창 닫기
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 창 닫기