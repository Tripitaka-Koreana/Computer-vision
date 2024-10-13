import cv2 as cv
import numpy as np

#%%

# 이미지 읽기
img = cv.imread('soccer.jpg')  # 'soccer.jpg' 이미지를 읽어옵니다.

# 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR 이미지를 그레이스케일로 변환합니다.

# Canny 엣지 감지
canny = cv.Canny(gray, 100, 200)  # Canny 엣지 감지를 수행합니다 (low_threshold=100, high_threshold=200).

#%%

# 윤곽선 찾기 hierarchy는 관계로 신경 ㄴㄴ
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)  # 윤곽선을 찾습니다.

# 첫 번째 윤곽선 출력
print(contour[0])  # 첫 번째 윤곽선을 출력합니다.
len(contour[0])  # 첫 번째 윤곽선의 포인트 수를 확인합니다.

# 윤곽선 필터링
lcontour = []  # 필터링된 윤곽선을 저장할 리스트
for i in range(len(contour)):
    if contour[i].shape[0] > 100:  # 윤곽선의 점 개수가 100개 초과인 경우
        lcontour.append(contour[i])  # 필터링된 윤곽선을 리스트에 추가합니다.

# 윤곽선 그리기
cv.drawContours(img, lcontour, -1, (0, 255, 0), 3)  # 필터링된 윤곽선을 원본 이미지에 초록색으로 그립니다.

# 결과 출력
cv.imshow('Original', img)  # 윤곽선이 그려진 원본 이미지를 출력합니다.
cv.waitKey()  # 키 입력을 기다립니다.

cv.imshow('Canny', canny)  # Canny 엣지 감지 결과를 출력합니다.
cv.waitKey()  # 키 입력을 기다립니다.

cv.destroyAllWindows()  # 모든 창을 닫습니다.