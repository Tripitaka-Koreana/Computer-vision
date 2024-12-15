import tensorflow as tf
import tensorflow.keras.datasets as ds
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
#%% Data handling
(x_train, y_train), (x_test, y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000, 784) #1차원으로 펼치는 과정
x_test=x_test.reshape(10000, 784)
x_train=x_train.astype(np.float32)/255.0  #0~1 normalization, 실수 연산을 위해 형 변환 후 정규화
x_test=x_test.astype(np.float32)/255.0 
y_train=tf.keras.utils.to_categorical(y_train, 10) #원핫코드로 변환
y_test=tf.keras.utils.to_categorical(y_test, 10)

#%% ANN model Construction
mlp=Sequential()  
#Dense: 완전 연결층, input_shape: 입력층 노드개수, units: 은닉층 512개 노드, tanh 활성 함수
mlp.add(Dense(units=512, activation="tanh", input_shape=(784,)))
# 출력층에 해당하는 완전 연결층, 텐서플로가 이전 층의 노드 개수를 알고 있어 input_shape 생략, softmax 활성 함수(출력층이 주로 사용)
mlp.add(Dense(units=10, activation="softmax")) 

#%% SGD
#compile과 fit 함수가 쌍으로 신경망 학습을 담당
#loss: 손실함수 - MSE: 평균제곱오차, optimizer: 최적화 방법 - SGD: 스토캐스틱 경사 하강법, 학습률 0.01
#metrics: 학습하는 도중에 성능 측정하는 기준 지시 - accuracy: 정확률을 기준
mlp.compile(loss="MSE", optimizer=SGD(learning_rate=0.01), metrics=["accuracy"]) #Learing environment configuration
#fit: 실제로 학습을 실행, 학습시간을 측정하는 시작점(time.time())
#epochs: 세대(50번 반복), batch_size: 미니배치크기, verbose: 학습 도중 세대마다 성능 출력(0,1,2)
mlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)
#예측을 통해 성능 측정 res[1]: 정확률(accuracy)
res=mlp.evaluate(x_test, y_test, verbose=0) #test_loss, test_accuracy
print("Acuuracy", res[1]*100) #정확률을 %로 표기

#%% Adam
from tensorflow.keras.optimizers import Adam
mlp.compile(loss="MSE", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]) #Learing environment configuration
mlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)
res=mlp.evaluate(x_test, y_test, verbose=0)
print("Acuuracy", res[1]*100)