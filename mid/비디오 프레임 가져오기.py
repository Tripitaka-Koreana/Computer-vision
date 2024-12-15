import cv2 as cv
import sys
import numpy as np

#%% 비디오 캡처 객체 생성
cap = cv.VideoCapture('slow_traffic_small.mp4')  # 비디오 파일 열기
if not cap.isOpened():  # 비디오가 정상적으로 열리지 않은 경우
    sys.exit("연결 실패")  # 프로그램 종료

frames = []  # 프레임을 저장할 리스트 초기화

while True:  # 비디오 프레임을 반복해서 처리
    ret, frame = cap.read()  # 프레임 읽기
    
    if not ret:  # 프레임을 읽지 못한 경우
        print("프레임 획득 실패")  # 오류 메시지 출력
        break  # 루프 종료
    
    cv.imshow('Video Display', frame)  # 현재 프레임을 화면에 표시
    
    frate = cap.get(cv.CAP_PROP_FPS)  # 비디오의 FPS (초당 프레임 수) 가져오기
    key = cv.waitKey(int(1000/frate))  # FPS에 맞춰 키 입력 대기

    key = cv.waitKey(1)  # 1ms 대기 (다른 키 입력 감지)
    if key == ord('q'):  # 'q' 키가 눌린 경우
        break  # 루프 종료

cap.release()  # 비디오 캡처 객체 해제
#print(frate)  # (주석 처리된 부분) FPS 출력
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기