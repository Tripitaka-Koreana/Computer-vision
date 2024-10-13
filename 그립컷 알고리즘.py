import numpy as np
import cv2 as cv

#%% 
# 이미지를 읽어옵니다.
img = cv.imread('soccer.jpg')
img_show = np.copy(img)  # 이미지의 깊은 복사 생성

# 마스크 초기화: 모든 픽셀을 배경으로 설정
mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:, :] = cv.GC_PR_BGD  # 초기 마스크는 잠재적 배경으로 설정

BrushSiz = 9  # 브러시 크기 설정
LColor, RColor = (255, 0, 0), (0, 0, 255)  # 왼쪽 클릭과 오른쪽 클릭 색상 설정

# 마우스 이벤트 처리 함수
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # 왼쪽 클릭
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)  # 왼쪽 브러시 색상으로 원 그리기
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)  # 마스크에 전경으로 설정
    elif event == cv.EVENT_RBUTTONDOWN:  # 오른쪽 클릭
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)  # 오른쪽 브러시 색상으로 원 그리기
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)  # 마스크에 배경으로 설정
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:  # 왼쪽 버튼 드래그
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)  # 왼쪽 브러시 색상으로 원 그리기
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)  # 마스크에 전경으로 설정
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:  # 오른쪽 버튼 드래그
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)  # 오른쪽 브러시 색상으로 원 그리기
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)  # 마스크에 배경으로 설정
        
    cv.imshow("Painting", img_show)  # 현재 그리기 상태를 표시

# 윈도우 생성 및 마우스 콜백 설정
cv.namedWindow("Painting")
cv.setMouseCallback("Painting", painting)

#%% 
while True:
    if cv.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
        break

# GrabCut 알고리즘을 위한 초기화
background = np.zeros((1, 65), np.float64)  # 배경 모델 초기화
foreground = np.zeros((1, 65), np.float64)  # 전경 모델 초기화

# GrabCut 알고리즘 실행
cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
# 마스크에서 배경으로 설정된 픽셀을 0으로, 나머지를 1로 설정
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
# 마스크를 사용하여 최종 이미지를 생성
grab = img * mask2[:, :, np.newaxis]  # 마스크 브로드캐스팅

# GrabCut 결과 이미지를 표시
cv.imshow("Grab Cut Image", grab)
cv.waitKey()

cv.destroyAllWindows()  # 모든 창 닫기