import cv2 as cv

# 이미지 읽기
img = cv.imread('soccer.jpg')  # 'soccer.jpg' 이미지를 읽어옵니다.

# 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR 이미지를 그레이스케일로 변환합니다.

# Canny 엣지 감지
canny1 = cv.Canny(gray, 50, 150)  # 첫 번째 Canny 엣지 감지 (low_threshold=50, high_threshold=150)
canny2 = cv.Canny(gray, 100, 200)  # 두 번째 Canny 엣지 감지 (low_threshold=100, high_threshold=200)

# 결과 출력
cv.imshow('Original', gray)  # 원본 그레이스케일 이미지를 출력합니다.
cv.waitKey()  # 키 입력을 기다립니다.

cv.imshow('Canny1', canny1)  # 첫 번째 Canny 엣지 감지 결과를 출력합니다.
cv.waitKey()  # 키 입력을 기다립니다.

cv.imshow('Canny2', canny2)  # 두 번째 Canny 엣지 감지 결과를 출력합니다.

cv.waitKey()  # 키 입력을 기다립니다.
cv.destroyAllWindows()  # 모든 창을 닫습니다.