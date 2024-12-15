import skimage
import numpy as np
import cv2 as cv
import time

#%% 
# skimage에서 커피 이미지 데이터를 가져옵니다.
img = skimage.data.coffee()
# 이미지를 OpenCV 포맷으로 변환하여 표시합니다.
cv.imshow("Coffee", cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.waitKey()  # 키 입력을 기다립니다.

# 분할 시작 시간 기록
start = time.time()
# SLIC 알고리즘을 사용하여 초픽셀 생성
slic = skimage.segmentation.slic(img, compactness=20, n_segments=600, start_label=1)  # compactness: 뭉침 정도
print(slic.shape)  # 생성된 초픽셀의 형태 출력

# 평균 색상을 기반으로 RAG (Region Adjacency Graph) 생성
g = skimage.graph.rag_mean_color(img, slic, mode='similarity')
# Normalized Cut을 사용하여 분할 수행
ncut = skimage.graph.cut_normalized(slic, g)
print(img.shape, '분할 소요 시간', time.time() - start)  # 분할 소요 시간 출력
print(np.min(ncut))  # Normalized Cut 레이블의 최소값 출력
print(np.max(ncut))  # Normalized Cut 레이블의 최대값 출력
print(ncut)  # Normalized Cut 레이블 배열 출력
print(ncut.shape)  # Normalized Cut 배열의 형태 출력

#영역의 개수 출력
print(np.unique(ncut))

# Normalized Cut 경계를 이미지에 마킹
marking = skimage.segmentation.mark_boundaries(img, ncut)
# 마킹된 이미지를 uint8 형식으로 변환
ncut_img = np.uint8(marking * 255)

# Normalized Cut 결과 이미지를 OpenCV 포맷으로 변환하여 표시
cv.imshow("Normalized Cut", cv.cvtColor(ncut_img, cv.COLOR_RGB2BGR))
cv.waitKey()  # 키 입력을 기다립니다.

cv.destroyAllWindows()  # 모든 창을 닫습니다.
