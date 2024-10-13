import cv2 as cv
import numpy as np
import sys
import math

# 이미지를 불러옵니다
img = cv.imread('checkerboard.jpg')

# 이미지가 없으면 종료합니다
if img is None:
    sys.exit('파일이 없습니다')

# 이미지의 높이와 너비를 가져옵니다
h, w = img.shape[:2]

# 선을 그릴 원본 이미지의 복사본을 생성합니다
hl_img = img.copy()
hlp_img = img.copy()

# 이미지를 그레이스케일로 변환합니다
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny 엣지 감지기로 엣지를 검출합니다
edges = cv.Canny(gray, 125, 350)

# Hough 변환을 사용하여 선을 검출합니다
lines = cv.HoughLines(edges, 1, math.pi / 180, 90)

# 검출된 선을 hl_img에 그립니다
for i in range(len(lines)):
    for rho, theta in lines[i]:
        tx, ty = np.cos(theta), np.sin(theta)  # 선의 방향을 계산합니다
        x0, y0 = tx * rho, ty * rho  # 선 위의 점을 계산합니다
        # 선의 끝점 좌표를 계산합니다
        x1 = int(x0 + w * (-ty))
        y1 = int(y0 + w * tx)
        x2 = int(x0 - w * (-ty))
        y2 = int(y0 - w * tx)
        cv.line(hl_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 선을 그립니다

#%% HoughLinesP를 사용하여 선을 검출합니다

# 이미지를 다시 그레이스케일로 변환합니다
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 125, 350)  # 엣지를 다시 검출합니다
lines = cv.HoughLinesP(edges, 1, math.pi / 360, 90, 100, 10)  # 확률적 Hough 변환을 사용합니다
print(len(lines))  # 검출된 선의 개수를 출력합니다

# 검출된 선을 hlp_img에 그립니다
for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv.line(hlp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 선을 그립니다
        
# 원본 이미지를 보여줍니다
cv.imshow('IMG', img) 
cv.waitKey()

# HoughLines로 검출된 선이 그려진 이미지를 보여줍니다
cv.imshow('IMG', hl_img) 
cv.waitKey()

# HoughLinesP로 검출된 선이 그려진 이미지를 보여줍니다
cv.imshow('IMG', hlp_img) 
cv.waitKey()

# 모든 창을 닫습니다
cv.destroyAllWindows()