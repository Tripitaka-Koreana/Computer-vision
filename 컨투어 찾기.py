import cv2 as cv
import numpy as np
#%%

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray, 100, 200)
#%%

contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# cv.RETR_LIST: 모든 윤곽선을 찾고, 계층 정보를 저장하지 않습니다. 즉, 부모-자식 관계를 고려하지 않고 모든 윤곽선을 단순히 나열합니다.
# cv.RETR_EXTERNAL: 외부 윤곽선만 찾습니다. 즉, 내부 윤곽선은 무시합니다.
# cv.RETR_TREE: 모든 윤곽선을 찾고, 계층 구조를 형성합니다. 부모-자식 관계를 저장합니

# cv.CHAIN_APPROX_NONE: 모든 윤곽선 점을 저장합니다. 즉, 윤곽선의 모든 점이 반환됩니다.
# cv.CHAIN_APPROX_SIMPLE: 윤곽선의 점을 단순화하여 저장합니다. 수직 및 수평 경계만 저장하고 대각선은 생략하여 메모리를 절약합니다.

print(contour[0])
len(contour[0])

lcontour=[ ]
for i in range(len(contour)):
    if contour[i].shape[0] >100:
        lcontour.append(contour[i])
        
cv.drawContours(img, lcontour, -1, (0,255,0), 3)

cv.imshow('Original', img)
cv.waitKey()

cv.imshow('Canny', canny)
cv.waitKey()

cv.destroyAllWindows()
