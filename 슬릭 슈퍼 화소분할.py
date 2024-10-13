import skimage
import numpy as np
import cv2 as cv

#%% 
# skimage 라이브러리에서 커피 이미지 데이터를 가져옵니다.
img = skimage.data.coffee()
# 이미지를 OpenCV 포맷으로 변환하여 표시합니다.
cv.imshow("Coffee", cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.waitKey()  # 키 입력을 기다립니다.

# SLIC (Simple Linear Iterative Clustering) 알고리즘을 사용하여 초픽셀(segmentation) 생성
slic1 = skimage.segmentation.slic(img, compactness=20, n_segments=600)  # compactness: 뭉침 정도
print(np.min(slic1))  # 초픽셀 레이블의 최소값 출력
print(np.max(slic1))  # 초픽셀 레이블의 최대값 출력
print(slic1)  # 초픽셀 레이블 배열 출력
print(slic1.shape)  # 배열의 형태 출력

# 초픽셀 경계 마킹
sp_img1 = skimage.segmentation.mark_boundaries(img, slic1)  # 초픽셀 경계를 이미지에 표시
print(np.min(sp_img1))  # 경계 마킹 이미지의 최소값 출력
print(np.max(sp_img1))  # 경계 마킹 이미지의 최대값 출력
print(sp_img1.shape)  # 경계 마킹 이미지의 형태 출력

# 경계 마킹 이미지를 uint8 형식으로 변환
sp_img1 = np.uint8(sp_img1 * 255)

# 두 번째 SLIC 적용
slic2 = skimage.segmentation.slic(img, compactness=40, n_segments=600)  # compactness 증가
print(np.min(slic2))  # 두 번째 초픽셀 레이블의 최소값 출력
print(np.max(slic2))  # 두 번째 초픽셀 레이블의 최대값 출력
sp_img2 = skimage.segmentation.mark_boundaries(img, slic2)  # 두 번째 초픽셀 경계 마킹
print(np.min(sp_img2))  # 두 번째 경계 마킹 이미지의 최소값 출력
print(np.max(sp_img2))  # 두 번째 경계 마킹 이미지의 최대값 출력
sp_img2 = np.uint8(sp_img2 * 255)  # 두 번째 경계 마킹 이미지를 uint8 형식으로 변환

# 두 개의 초픽셀 마킹 이미지를 OpenCV 포맷으로 변환하여 표시
cv.imshow("Super Pixels, Com=20", cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR))  # compactness=20
cv.imshow("Super Pixels, Com=40", cv.cvtColor(sp_img2, cv.COLOR_RGB2BGR))  # compactness=40
cv.waitKey()  # 키 입력을 기다립니다.

cv.destroyAllWindows()  # 모든 창을 닫습니다.