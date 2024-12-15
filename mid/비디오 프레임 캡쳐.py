import cv2 as cv
import sys
import numpy as np

#%% 비디오 캡처 객체 생성
cap = cv.VideoCapture('slow_traffic_small.mp4')  # 비디오 파일 열기
if not cap.isOpened():  # 비디오가 정상적으로 열리지 않은 경우
    sys.exit("연결 실패")  # 프로그램 종료
    
frames = []  # 선택한 프레임을 저장할 리스트 초기화

while True:  # 비디오 프레임을 반복해서 처리
    ret, frame = cap.read()  # 프레임 읽기
    
    if not ret:  # 프레임을 읽지 못한 경우
        print("프레임 획득 실패")  # 오류 메시지 출력
        break  # 루프 종료
    
    cv.imshow('Video Display', frame)  # 현재 프레임을 화면에 표시
    
    key = cv.waitKey(1)  # 1ms 대기하며 키 입력 감지
    if key == ord('c'):  # 'c' 키가 눌린 경우
        frames.append(frame)  # 현재 프레임을 리스트에 추가
    elif key == ord('q'):  # 'q' 키가 눌린 경우
        break  # 루프 종료

cap.release()  # 비디오 캡처 객체 해제
cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

if len(frames) > 0:  # 선택한 프레임이 있는 경우
    imgs = frames[0]  # 첫 번째 프레임을 초기 이미지로 설정
    for i in range(1, min(3, len(frames))):  # 최대 3개의 프레임을 수평으로 결합
        imgs = np.hstack((imgs, frames[i]))  # 프레임들을 수평으로 결합
        
    cv.imshow('collected images', imgs)  # 결합된 이미지를 화면에 표시
    
    cv.waitKey()  # 키 입력 대기
    cv.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기