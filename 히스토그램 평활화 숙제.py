import cv2 as cv
import matplotlib.pyplot as plt
import sys
import numpy as np

# 이미지 불러오기
img = cv.imread('peppers.bmp')  # 이미지를 읽어옵니다.

if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 종료합니다.

print(img.shape)  # 원본 이미지의 크기 출력

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
print(img_rgb.shape)  # RGB 이미지의 크기 출력

ycbcr_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)  # BGR에서 YCbCr 색상 공간으로 변환
print(ycbcr_img.shape)  # YCbCr 이미지의 크기 출력

channel_y = ycbcr_img[:, :, 0]  # Y 성분 추출
channel_y_his = cv.calcHist([channel_y], [0], None, [256], [0, 256])  # Y 성분의 히스토그램 계산

channel_y_eq = cv.equalizeHist(channel_y)  # Y 성분에 히스토그램 균등화 적용
channel_y_eq_his = cv.calcHist([channel_y_eq], [0], None, [256], [0, 256])  # 균등화 후 Y 성분의 히스토그램 계산
print(channel_y_eq_his[250]) #균등화된 히스토그램에서 250값을 가진 화소의 개수 출력

ycbcr_img[:, :, 0] = channel_y_eq  # 균등화된 Y 성분을 YCbCr 이미지에 다시 넣기

equalized_image_rcbcr = cv.cvtColor(ycbcr_img, cv.COLOR_YCrCb2BGR)  # YCbCr 이미지를 BGR로 변환

# RGB의 각 채널에 대해 히스토그램 균등화 수행
r_channel = img_rgb[:, :, 0]  # R 채널 추출
g_channel = img_rgb[:, :, 1]  # G 채널 추출
b_channel = img_rgb[:, :, 2]  # B 채널 추출

r_eq = cv.equalizeHist(r_channel)  # R 채널에 히스토그램 균등화 적용
g_eq = cv.equalizeHist(g_channel)  # G 채널에 히스토그램 균등화 적용
b_eq = cv.equalizeHist(b_channel)  # B 채널에 히스토그램 균등화 적용

# 균등화된 채널을 병합하여 결과 이미지 생성
equalized_image_rgb = cv.merge((r_eq, g_eq, b_eq))  # R, G, B 채널을 병합

# 평활화 결과 출력
plt.figure(1)

plt.subplot(1, 2, 1)
plt.title('Original Y')  # 원본 Y 성분 히스토그램 제목
plt.plot(channel_y_his, color='r', linewidth=1)  # 원본 Y 히스토그램 그리기

plt.subplot(1, 2, 2)
plt.title('Equalized Y')  # 균등화된 Y 성분 히스토그램 제목
plt.plot(channel_y_eq_his, color='r', linewidth=1)  # 균등화된 Y 히스토그램 그리기

plt.tight_layout()  # 서브플롯 간의 간격 조정
plt.show()  # 히스토그램 표시

# 이미지 출력
plt.figure(2)

plt.subplot(1, 3, 1)
plt.title('Original Image')  # 원본 이미지 제목
plt.imshow(img_rgb)  # RGB로 변환된 원본 이미지 표시
plt.axis('off')  # 축 표시 끄기

plt.subplot(1, 3, 2)
plt.title('Equalized Y')  # Y 균등화 이미지 제목
plt.imshow(cv.cvtColor(equalized_image_rcbcr, cv.COLOR_BGR2RGB))  # RGB로 변환된 Y 균등화 이미지 표시
plt.axis('off')  # 축 표시 끄기

plt.subplot(1, 3, 3)
plt.title('Equalized RGB')  # RGB 균등화 이미지 제목
plt.imshow(equalized_image_rgb)  # RGB 균등화된 이미지 표시
plt.axis('off')  # 축 표시 끄기

# 원본 이미지, Y 균등화 이미지, RGB 균등화 이미지를 수평으로 결합하여 표시
combined_image = np.hstack((img_rgb, cv.cvtColor(equalized_image_rcbcr, cv.COLOR_BGR2RGB), equalized_image_rgb))
cv.imshow('Equalized', cv.cvtColor(combined_image, cv.COLOR_RGB2BGR))  # 결합된 이미지를 BGR로 변환 후 표시
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()