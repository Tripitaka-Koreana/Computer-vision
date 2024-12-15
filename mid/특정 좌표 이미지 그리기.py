import cv2 as cv
import sys

# 이미지 파일 읽기
img = cv.imread('girl_laughing.jpg')  

# 이미지 파일이 존재하지 않을 경우 프로그램 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')  # 오류 메시지 출력 및 종료

# 이미지에 직사각형 그리기: (왼쪽 상단 좌표, 오른쪽 하단 좌표, 색상(BGR), 두께)
cv.rectangle(img, (830, 30), (1000, 200), (0, 0, 255), 2)  # 빨간색 직사각형

# 이미지에 텍스트 추가: (텍스트, 위치, 폰트, 크기, 색상(BGR), 두께)
cv.putText(img, 'laugh', (830, 24), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 파란색 글씨

# 수정된 이미지를 화면에 표시
cv.imshow('Draw', img)

# 키 입력을 대기하고 모든 창 닫기
cv.waitKey()  # 키 입력 대기
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기