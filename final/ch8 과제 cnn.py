import tensorflow as tf
import tensorflow.keras.datasets as ds
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

#%% Data
(x_train, y_train), (x_test, y_test) = ds.fashion_mnist.load_data()

# 클래스 이름 목록
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%% 
# 5장의 사진 출력
plt.figure(figsize=(20, 4))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    # 이미지를 2D 배열로 reshape
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])  # 해당 클래스 이름 출력
    plt.axis('off')  # 축 표시 제거
plt.show()

#%% Data handling
# 데이터 확인
print("학습 데이터: ")
print(x_train.shape, y_train.shape)
print("테스트 데이터: ")
print(x_test.shape, y_test.shape)
print()

# 각 이미지를 28x28 형태에서 784 크기의 1차원 배열로 변경
x_train = x_train.reshape(60000, 28, 28, 1)  # 훈련 데이터의 형태를 (60000, 28 x 28)로 변환
x_test = x_test.reshape(10000, 28, 28, 1)    # 테스트 데이터의 형태를 (10000, 28 x 28)로 변환

# 이미지 데이터를 0-255 범위에서 0-1 범위로 정규화
x_train = x_train.astype(np.float32) / 255.0  # 훈련 데이터 정규화
x_test = x_test.astype(np.float32) / 255.0   # 테스트 데이터 정규화

# 레이블을 원-핫 인코딩 (10개의 클래스)
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 훈련 레이블 원-핫 인코딩
y_test = tf.keras.utils.to_categorical(y_test, 10)    # 테스트 레이블 원-핫 인코딩
#%% ANN model Construction
# 모델 구축
cnn = Sequential()  # Sequential 모델 생성

# 첫 번째 은닉층 (입력층과 연결됨)
cnn.add(Conv2D(64, (3,3), activation="relu", padding='same', input_shape=(28, 28, 1)))  # 784개의 입력을 받는 은닉층, 1024개의 노드, ReLU 활성화 함수

# 두 번째 은닉층
cnn.add(Conv2D(32, (3,3), activation="relu", padding='valid'))  # 784개의 입력을 받는 은닉층, 1024개의 노드, ReLU 활성화 함수

# 풀링층
cnn.add(MaxPooling2D(pool_size= (2,2), strides=2))

# 1차원으로 바꿔서 넣어줌
cnn.add(Flatten())

# 100개의 노드 FC
cnn.add(Dense(units=100, activation="relu"))

# 출력층 (10개의 클래스에 대한 확률을 출력)
cnn.add(Dense(units=10, activation="softmax"))  # 10개의 클래스를 분류하기 위한 출력층, Softmax 활성화 함수

#%% Adam
# 모델 컴파일
#cnn.compile(loss="categorical_crossentropy",  # 손실 함수는 다중 클래스 분류 문제에 적합한 categorical_crossentropy 사용
#            optimizer=Adam(learning_rate=0.0001),  # Adam 옵티마이저 사용, 학습률은 0.0001로 설정
#            metrics=["accuracy"])  # 평가 지표로 정확도를 사용

# 모델 학습
#hist = cnn.fit(x_train, y_train, batch_size=128, epochs=30,  # 배치 크기 128, 50 에포크 동안 학습
#               validation_data=(x_test, y_test),  # 검증 데이터셋 제공
#               verbose=2)  # 학습 과정을 출력 x


cnn.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])
hist = cnn.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

# 모델 평가
print("Accuracy: ", cnn.evaluate(x_test, y_test, verbose=0)[1] * 100)  # 테스트 데이터에서 정확도 출력

# 모델 저장
cnn.save("fashion_mnist_cnn.h5")

#%% Accuracy
# 학습 과정에서의 정확도 변화 시각화
dmlp=tf.keras.models.load_model('fashion_mnist_cnn.h5')

plt.plot(hist.history["accuracy"])  # 훈련 데이터의 정확도 그래프
plt.plot(hist.history["val_accuracy"])  # 검증 데이터의 정확도 그래프
plt.title("Accuracy Graph")  # 그래프 제목 설정
plt.xlabel("epocs")  # x축 레이블 설정 (에포크 수)
plt.ylabel("accuracy")  # y축 레이블 설정 (정확도)
plt.legend(["train", "test"])  # 범례 설정
plt.grid()  # 그리드 추가
plt.show()  # 그래프 출력

#%% Loss
# 학습 과정에서의 손실 함수 값 변화 시각화

plt.plot(hist.history["loss"])  # 훈련 데이터의 손실 함수 값 그래프
plt.plot(hist.history["val_loss"])  # 검증 데이터의 손실 함수 값 그래프
plt.title("Loss Graph")  # 그래프 제목 설정
plt.xlabel("epocs")  # x축 레이블 설정 (에포크 수)
plt.ylabel("Loss")  # y축 레이블 설정 (손실 값)
plt.legend(["train", "test"])  # 범례 설정
plt.grid()  # 그리드 추가
plt.show()  # 그래프 출력


#%% 
# 신경망 모델 평가
dmlp=tf.keras.models.load_model('fashion_mnist_cnn.h5')

train_res = dmlp.evaluate(x_train, y_train, verbose=2)
test_res = dmlp.evaluate(x_test, y_test, verbose=2)

print(f"훈련 데이터 손실: {train_res[0]}, 훈련 데이터 정확도: {train_res[1]}")
print(f"검증 데이터 손실: {test_res[0]}, 검증 데이터 정확도: {test_res[1]}")
print()

#%% 
# 예측 결과 (각 클래스에 대한 확률) (1, 784) 형태로 변경해야함
i = 5 #i번째 이미지
res = dmlp.predict(np.expand_dims(x_train[i], axis=0))  # 첫 번째 이미지에 대한 예측

# 예측된 확률 출력 (각 클래스에 대한 확률)
predicted_probabilities = res[i]  # 첫 번째 이미지 예측 결과

# 예측된 클래스 확률 출력
print()
print(str(i) + "번 이미지에 대한 예측 확률:")
for i, prob in enumerate(predicted_probabilities):
    # 클래스 이름과 확률을 출력
    print(f"{class_names[i]}: {prob*100:.2f}% [{prob}]")  # 확률을 백분율로 출력 (소수점 두 자리)

# 예측된 클래스 출력
predicted_class = np.argmax(predicted_probabilities)  # 가장 높은 확률을 가진 클래스
print(f"\n예측된 클래스: {class_names[predicted_class]}")  # 예측된 클래스 이름 출력
