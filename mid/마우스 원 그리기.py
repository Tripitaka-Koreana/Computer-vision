import cv2 as cv 
import sys

# 이미지 파일 읽기
img = cv.imread('soccer.jpg')

# 이미지 파일이 존재하지 않을 경우 프로그램 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')  # 오류 메시지 출력 및 종료

BrushSiz = 5  # 붓의 크기
LColor, RColor = (255, 0, 0), (0, 0, 255)  # 파란색과 빨간색 정의

# 마우스 이벤트 처리 함수 정의
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:   
        cv.circle(img, (x, y), BrushSiz, LColor, -1)  # 마우스 왼쪽 버튼 클릭하면 파란색 원 그리기
    elif event == cv.EVENT_RBUTTONDOWN: 
        cv.circle(img, (x, y), BrushSiz, RColor, -1)  # 마우스 오른쪽 버튼 클릭하면 빨간색 원 그리기
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)  # 왼쪽 버튼 클릭 상태에서 이동하면 파란색 원 그리기
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x, y), BrushSiz, RColor, -1)  # 오른쪽 버튼 클릭 상태에서 이동하면 빨간색 원 그리기

    cv.imshow('Painting', img)  # 수정된 이미지를 다시 표시

# 'Painting'이라는 이름의 윈도우 생성
cv.namedWindow('Painting')
cv.imshow('Painting', img)  # 원본 이미지 표시

# 마우스 콜백 함수 설정
cv.setMouseCallback('Painting', painting)

# 무한 루프: 키 입력 감지
while True:
    if cv.waitKey(1) == ord('q'):  # 'q' 키가 눌리면 종료
        cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기
        break