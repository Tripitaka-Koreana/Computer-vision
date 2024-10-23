import cv2 as cv
import numpy as np

#%% 이미지 배열 생성
# 10x10 크기의 이진 이미지 생성 (0은 검정색, 1은 흰색)
img = np.array([[0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,1,1,0,0,0,0,0],
                 [0,0,0,1,1,1,0,0,0,0],
                 [0,0,0,1,1,1,1,0,0,0],
                 [0,0,0,1,1,1,1,1,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]], dtype=np.float32)

# x 방향과 y 방향의 Sobel 필터 생성
ux = np.array([[-1, 0, 1]])  # x 방향 필터
uy = np.array([-1, 0, 1]).transpose()  # y 방향 필터

# 가우시안 커널 생성
k = cv.getGaussianKernel(3, 1)
print(k.shape)  # 커널의 크기 출력
g = np.outer(k, k.transpose())  # 2D 가우시안 커널 생성

#%% 이미지의 기울기 계산
dy = cv.filter2D(img, cv.CV_32F, uy)  # y 방향 기울기
dx = cv.filter2D(img, cv.CV_32F, ux)  # x 방향 기울기

# 각 방향의 기울기 제곱 계산
dxx = dx * dx  # x 방향 기울기의 제곱
dyy = dy * dy  # y 방향 기울기의 제곱
dyx = dy * dx  # xy 방향 기울기

# 가우시안 블러 적용
gdxx = cv.filter2D(dxx, cv.CV_32F, g)  # x 방향 기울기의 제곱에 가우시안 블러 적용
gdyy = cv.filter2D(dyy, cv.CV_32F, g)  # y 방향 기울기의 제곱에 가우시안 블러 적용
gdyx = cv.filter2D(dyx, cv.CV_32F, g)  # xy 방향 기울기에 가우시안 블러 적용

# Harris 코너 검출기 계산
C = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdxx + gdyy) * (gdxx + gdyy)

# 코너 검출 후, 이미지 업데이트
for j in range(1, C.shape[0]-1):
    for i in range(1, C.shape[1]-1):
        # 조건에 맞는 점을 코너로 판단하고 img 배열에서 해당 픽셀을 9로 설정
        if C[j, i] > 0.1 and sum(sum(C[j, i] > C[j-1:j+2, i-1:i+2])) == 8:
            img[j, i] = 9

# numpy 출력 옵션 설정
np.set_printoptions(precision=2)

# 각 중간 결과 출력
print(dy)  # y 방향 기울기
print(dx)  # x 방향 기울기
print(dyy)  # y 방향 기울기의 제곱
print(dxx)  # x 방향 기울기의 제곱
print(dyx)  # xy 방향 기울기
print(gdyy)  # y 방향 기울기에 가우시안 블러 적용 후 결과
print(gdxx)  # x 방향 기울기에 가우시안 블러 적용 후 결과
print(gdyx)  # xy 방향 기울기에 가우시안 블러 적용 후 결과
print(C)  # Harris 코너 검출 결과
print(img)  # 최종 이미지

#%% 최종 결과를 저장할 배열 생성
poping = np.zeros([160, 160], np.uint8)

# 각 픽셀에 대해 C 값에 기반하여 poping 배열을 업데이트
for j in range(0, 160):
    for i in range(0, 160):
        poping[j, i] = np.uint8((C[j // 16, i // 16] + 0.06) * 700)

# 이미지 출력
cv.imshow('Img Display', poping)        
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 종료
