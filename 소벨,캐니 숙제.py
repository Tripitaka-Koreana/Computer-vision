import cv2 as cv
import numpy as np

img=cv.imread('food.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 소벨 필터 작용
grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

# (30, 40) 위치의 dx, dy
x, y = 40, 30
dx = sobel_x[y, x]
dy = sobel_y[y, x]

# dx와 dy를 배열로 변환
dx_array = np.array([[dx]], dtype=np.float32)
dy_array = np.array([[dy]], dtype=np.float32)

# 에지 강도,  그레디언트 방향
magnitude, angle = cv.cartToPolar(dx_array, dy_array, angleInDegrees=True)
angle_radians = np.radians(angle)

# 에지 강도 맵
edge_strength=cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

cv.imshow('Original', gray)
cv.waitKey()

cv.imshow('sobel_x', sobel_x)
cv.waitKey()
cv.imshow('sobel_y', sobel_y)
cv.waitKey()
cv.imshow('edge_strength', edge_strength)
cv.waitKey()

print(f"Coordinates (x, y): ({x}, {y})")
print(f"dx: {dx}, dy: {dy}")
print(f"Edge Strength (Magnitude): {magnitude}")
print(f"Gradient Direction (Angle): {angle}")
print(f"Gradient Direction (angle_radians): {angle_radians}")

cv.destroyAllWindows()
#%%

# 가우시안 블러 적용
blurred = cv.GaussianBlur(gray, (3, 3), 1, 1)

# Canny 엣지 검출
canny1 = cv.Canny(blurred, 50, 150)
canny2 = cv.Canny(blurred, 100, 200)

# 결과 표시
cv.imshow('blurred', blurred)
cv.waitKey()
cv.imshow('canny1', canny1)
cv.waitKey()
cv.imshow('canny2', canny2)
cv.waitKey()

cv.destroyAllWindows()

#%%

