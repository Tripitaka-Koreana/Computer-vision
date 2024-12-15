import cv2 as cv

# 이미지 읽기
img = cv.imread('soccer.jpg')  # 'soccer.jpg' 이미지를 읽어옵니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환합니다.

# Sobel 필터를 사용하여 x 방향과 y 방향의 기울기를 계산합니다.
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)  # x 방향 기울기
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)  # y 방향 기울기

# 기울기 값을 절대값으로 변환하여 8비트로 변환합니다.
sobel_x = cv.convertScaleAbs(grad_x)  # x 방향 기울기를 절대값으로 변환
sobel_y = cv.convertScaleAbs(grad_y)  # y 방향 기울기를 절대값으로 변환

# x 방향과 y 방향의 기울기를 결합하여 엣지 강도를 계산합니다.
edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)  # 두 기울기를 합성

#%% 결과 출력
cv.imshow('Original', gray)  # 원본 그레이스케일 이미지를 출력합니다.
cv.waitKey()  # 키 입력을 기다립니다.

cv.imshow('sobel_x', sobel_x)  # x 방향 기울기 이미지를 출력합니다.
cv.waitKey()  # 키 입력을 기다립니다.

cv.imshow('sobel_y', sobel_y)  # y 방향 기울기 이미지를 출력합니다.
cv.waitKey()  # 키 입력을 기다립니다.

cv.imshow('sobel_strength', edge_strength)  # 엣지 강도 이미지를 출력합니다.

cv.waitKey()  # 키 입력을 기다립니다.
cv.destroyAllWindows()  # 모든 창을 닫습니다.
