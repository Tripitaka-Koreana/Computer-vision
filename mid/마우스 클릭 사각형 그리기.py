import cv2 as cv
import sys

#%% 이미지 파일 읽기
img = cv.imread('girl_laughing.jpg')  

# 이미지 파일이 존재하지 않을 경우 프로그램 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')  # 오류 메시지 출력 및 종료

# 마우스 이벤트 처리 함수 정의
def draw(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 클릭 시
        cv.rectangle(img, (x, y), (x + 200, y + 200), (0, 0, 255), 2)  # 빨간색 직사각형 그리기
    elif event == cv.EVENT_RBUTTONDOWN:  # 오른쪽 마우스 버튼 클릭 시
        cv.rectangle(img, (x, y), (x + 100, y + 100), (255, 0, 0), 2)  # 파란색 직사각형 그리기

    cv.imshow('Drawing', img)  # 수정된 이미지 표시

# 'Drawing'이라는 이름의 윈도우 생성
cv.namedWindow('Drawing')
cv.imshow('Drawing', img)  # 원본 이미지 표시

# 마우스 콜백 함수 설정
cv.setMouseCallback('Drawing', draw)

# 무한 루프: 키 입력 감지
while True:
    if cv.waitKey(1) == ord('q'):  # 'q' 키가 눌리면 종료
        cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기
        break