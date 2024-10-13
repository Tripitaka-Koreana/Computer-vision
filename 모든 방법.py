import cv2 as cv
import sys
import numpy as np

#%% 이미지 읽기
img = cv.imread("soccer.jpg")  # 이미지를 파일에서 읽어옵니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환합니다.
if img is None:
    sys.exit("No File exists.")  # 이미지가 존재하지 않으면 프로그램을 종료합니다.
    
#%% 마우스 이벤트
img_mouse = img.copy()  # 또는 img_copy = img[:]를 사용해도 됩니다.
lx, ly = 0, 0
BrushSiz = 5  # 붓의 크기

# 마우스 이벤트 처리 함수 정의
def draw(event, x, y, flags, param):
    global lx, ly
    if event == cv.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 시
        lx, ly = x, y  # 시작 좌표 저장
    elif event == cv.EVENT_LBUTTONUP:  # 왼쪽 버튼 떼기 시
        cv.rectangle(img_mouse, (lx, ly), (x, y), (0, 255, 0), 2)  # 2는 두깨 사각형 그리기
    elif event == cv.EVENT_RBUTTONDOWN:  # 오른쪽 마우스 버튼 클릭 시 
        cv.rectangle(img_mouse, (x, y), (x + 100, y + 100), (255, 0, 0), 2)  # 파란색 직사각형
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_mouse, (x, y), BrushSiz, (255, 0, 0), -1)  # 왼쪽 버튼 클릭 상태에서 이동하면 파란색 원 그리기

    cv.imshow('Drawing', img_mouse)  # 수정된 이미지 표시

# 'Drawing'이라는 이름의 윈도우 생성
cv.namedWindow('Drawing')
cv.imshow('Drawing', img_mouse)  # 원본 이미지 표시

# 마우스 콜백 함수 설정
cv.setMouseCallback('Drawing', draw)

# 무한 루프: 키 입력 감지
while True:
    if cv.waitKey(1) == ord('q'):  # 'q' 키가 눌리면 종료
        cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기
        break

#%% 이미지 크기 조정 (25%로 축소)
resize_img = cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)

#%% 이미지 감마 조정 사용자 정의 함수
# 감마 조정 함수 정의
def gamma(f, gamma=1.0):
    f1 = f / 255.0  # 픽셀 값을 [0, 1] 범위로 정규화
    return np.uint8(255 * (f1 ** gamma))  # 감마 조정을 적용한 후 다시 [0, 255] 범위로 변환

# 다양한 감마 값을 적용하여 이미지를 나란히 배치
gc = np.hstack((
    gamma(resize_img, 0.5),  # 감마 0.5
    gamma(resize_img, 0.75), # 감마 0.75
    gamma(resize_img, 1.0),  # 감마 1.0 (원본 이미지)
    gamma(resize_img, 2.0),  # 감마 2.0
    gamma(resize_img, 3.0)   # 감마 3.0
))    

# 감마 조정 결과를 표시
cv.imshow("gamma", gc)  # 감마 조정된 이미지를 나란히 표시

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

#%% R 채널에 대해 이진화
# Otsu의 이진화 방법을 사용하여 R 채널을 이진화합니다.
# cv.THRESH_BINARY | cv.THRESH_OTSU를 사용하여 이진화 임계값을 자동으로 계산합니다.
# threshold(gray 이미지, 임계값, 임계값보다 크면 되는 값, type)
#_, binary_image = cv.threshold(img, 127, 255, cv.THRESH_BINARY) 이러면 127이상은 다 255로
# 여기는 오츄( 알아서 최적의 값 찾아줌 )
t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# 계산된 임계값 출력
print("threshold", t)  # 사용된 임계값 출력

# R 채널과 이진화된 결과를 각각 표시
cv.imshow('R', img[:,:,2])  # 원본 R 채널 이미지 표시
cv.imshow('R binary', bin)    # 이진화된 R 채널 이미지 표시

cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

#%% BGR에서 HSV로 변환해서 특정 색상 검출 채널 2개
# cv.cvtColor(img, cv.COLOR_BGR2HSV, dst=dst_hsv) 랑 dst_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 같음
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # BGR 색상 공간을 HSV 색상 공간으로 변환

mask1 = cv.inRange(hsv, (90, 25, 25), (135, 160, 255))
cv.imshow("mask1", mask1)
mask2 = cv.inRange(hsv, (0,25, 25),(30,160,255))
cv.imshow("mask2", mask2)
mask3 = cv.inRange(hsv, (150,25, 25),(180,160,255))
cv.imshow("mask3", mask3)

mask = mask1 | mask2 | mask3
cv.imshow("Mask", mask)  # 생성된 마스크 이미지 보여주기
cv.waitKey()  # 키 입력 대기


dst = np.full(img.shape, 255, dtype=np.uint8)  # 흰색 배경으로 채운 배열 생성

# 마스크를 사용하여 원본 이미지에서 검출된 부분만 가져오기
detected = cv.copyTo(img, mask, dst)  # 마스크를 적용하여 원본 이미지에서 검출된 부분만 dst에 복사
cv.imshow("Result", detected)  # 최종 결과 이미지 보여주기
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기




