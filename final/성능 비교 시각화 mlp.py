# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:01:17 2023

@author: BigData
"""

import tensorflow as tf
import tensorflow.keras.datasets as ds
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
#%% Data handling
(x_train, y_train), (x_test, y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000, 784)
x_test=x_test.reshape(10000, 784)
x_train=x_train.astype(np.float32)/255.0  #0~1 normalization
x_test=x_test.astype(np.float32)/255.0 
y_train=tf.keras.utils.to_categorical(y_train, 10)
y_test=tf.keras.utils.to_categorical(y_test, 10)

#%% ANN model Construction
mlp_sgd=Sequential()
mlp_sgd.add(Dense(units=512, activation="tanh", input_shape=(784,)))
mlp_sgd.add(Dense(units=10, activation="softmax"))

#%% SGD
mlp_sgd.compile(loss="MSE", optimizer=SGD(learning_rate=0.01), metrics=["accuracy"]) #Learing environment configuration
hist_sgd=mlp_sgd.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)
#print(hist_sgd.history)
#hist_sgd.history: 학습 결과를 저장한 딕셔너리 - 키마다 epochs 크기의 리스트 저장
#lost: 훈련집합의 손실 함수값, accuracy:훈련 집합의 정확률 , val_loss: 검증 집합의 손실 함숫값 , val_accuracy: 검증 집합의 정확률
print("Acuuracy", mlp_sgd.evaluate(x_test, y_test, verbose=0)[1]*100)

#%% Adam
mlp_adam=Sequential()
mlp_adam.add(Dense(units=512, activation="tanh", input_shape=(784,)))
mlp_adam.add(Dense(units=10, activation="softmax"))
mlp_adam.compile(loss="MSE", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]) #Learing environment configuration
hist_adam=mlp_adam.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

print("Acuuracy", mlp_adam.evaluate(x_test, y_test, verbose=0)[1]*100)

#%%
import matplotlib.pyplot as plt

plt.plot(hist_sgd.history["accuracy"], "r--") #default color: cyan
plt.plot(hist_sgd.history["val_accuracy"], "r")
plt.plot(hist_adam.history["accuracy"], "b--")
plt.plot(hist_adam.history["val_accuracy"], "b")
plt.title("Comparison of SGD and Adam")
plt.ylim(0.7,1.0)
plt.xlabel("epocs")
plt.ylabel("accuracy")
plt.legend(["train_sgd", "test_sgd", "train_adam", "test_adam"])
plt.grid()
plt.show()
