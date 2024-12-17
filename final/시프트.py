#des.shape
#Out[17]: (4415, 128) 4415개의 keypoint, keypoint하나당 128차원의 배열(4x4x8(히스토그램))
#cv.SIFT_create(특징점 개수, 옥타브 개수, 테일러 확장, 에지 특징점 필터, 옥타브0의 가우시안 표준편차)
#sift=cv.SIFT_create(500) : 500개의 특징점 검출하기

import cv2 as cv

img=cv.imread("mot_color83.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("Gray shape: ", gray.shape)

sift=cv.SIFT_create()

kp, des = sift.detectAndCompute(gray, None)
print("특징점 개수: " + str(len(kp)))
i = 5
print(str(i) + "번 좌표 (x,y): " + str(kp[i].pt))
print(str(i) + "번 방향 : " + str(kp[i].angle))
print(str(i) + "번 크기 : " + str(kp[i].size))
print(str(i) + "번 중요 : " + str(kp[i].response))
print(str(i) + "번 des : " + str(des[i]))

#print("Keypoints:\n", cv.KeyPoint_convert(kp))
#kp1=cv.KeyPoint_convert(kp) #- 특징점을 x,y 좌표로 표현

#kp1[0]
#Out[11]: array([  2.3539052, 530.67145  ], dtype=float32)

gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT", gray)

cv.waitKey()
cv.destroyAllWindows()