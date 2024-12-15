#kp[0].size
#Out[9]: 1.8633222579956055
#kp[0].response
#Out[13]: 0.022232957184314728

#len(des)
#Out[14]: 4415
#type(des)
#Out[15]: numpy.ndarray
#des.shape
#Out[17]: (4415, 128) 4415개의 keypoint, keypoint하나당 128차원의 배열
import cv2 as cv

img=cv.imread("mot_color70.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("Gray shape: ", gray.shape)

#cv.SIFT_create(특징점 개수, 옥타브 개수, 테일러 확장, 에지 특징점 필터, 옥타브0의 가우시안 표준편차)
sift=cv.SIFT_create()
#sift=cv.SIFT_create(500) : 500개의 특징점 검출하기

#kp = 특징점, des = 기술자
kp, des = sift.detectAndCompute(gray, None)
# len(kp) - 특징점 개수 출력 
print("Keypoints:\n", cv.KeyPoint_convert(kp))
kp1=cv.KeyPoint_convert(kp) #- 특징점을 x,y 좌표로 표현

#kp1[0]
#Out[11]: array([  2.3539052, 530.67145  ], dtype=float32)
print("Number of Keypointd: ", len(kp))




gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT", gray)

cv.waitKey()
cv.destroyAllWindows()