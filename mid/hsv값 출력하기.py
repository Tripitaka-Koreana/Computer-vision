import cv2 as cv
import sys
import numpy as np

# 마우스 클릭 이벤트 처리 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # 클릭한 좌표의 HSV 값 출력
        hsv_value = hsv[y, x]  # y, x 순서로 접근
        print(f"HSV at ({x}, {y}): {hsv_value}")

# 이미지 읽기
img = cv.imread('BALLOON.BMP')
if img is None:
    sys.exit("No File exists.")

# BGR => HSV로 변환 (색상, 채도, 밝기)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 마우스 클릭 이벤트 설정
cv.namedWindow("Image")
cv.setMouseCallback("Image", mouse_callback)

while True:
    # 결과 이미지 표시
    cv.imshow("Image", img)
    
    # 키 입력 대기
    key = cv.waitKey(1)
    if key == 27:  # ESC 키
        break
cv.destroyAllWindows()