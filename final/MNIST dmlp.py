import tensorflow as tf
import tensorflow.keras.datasets as ds
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#%% Data handling
(x_train, y_train), (x_test, y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000, 784)
x_test=x_test.reshape(10000, 784)
x_train=x_train.astype(np.float32)/255.0  #0~1 normalization
x_test=x_test.astype(np.float32)/255.0 
y_train=tf.keras.utils.to_categorical(y_train, 10)
y_test=tf.keras.utils.to_categorical(y_test, 10)

#%% ANN model Construction
#깊은 다층 퍼셉트론, 3개의 은닉층
dmlp=Sequential()
dmlp.add(Dense(units=1024, activation="relu", input_shape=(784,)))
dmlp.add(Dense(units=512, activation="relu"))
dmlp.add(Dense(units=512, activation="relu"))
dmlp.add(Dense(units=10, activation="softmax"))

#%% Adam
#categorical_crossentropy: 교차 엔트로피를 손실함수로 사용
dmlp.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), 
             metrics=["accuracy"]) #Learing environment configuration
hist=dmlp.fit(x_train, y_train, batch_size=128, epochs=50, 
         validation_data=(x_test, y_test), verbose=2)

print("Acuuracy", dmlp.evaluate(x_test, y_test, verbose=0)[1]*100)

#fit로 학습을 마친 신경망 모델의 구조 정보와 가중치 값을 지정한 파일에 저장하는 함수
#필요할 때 load_model 함수로 불러서 사용
dmlp.save("dmlp_trained.h5")
#dmlp.load_model("dmlp_trained.h5")

#%% Accuracy
import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"]) #default color: cyan
plt.plot(hist.history["val_accuracy"])
plt.title("Accuracy Graph")
plt.xlabel("epocs")
plt.ylabel("accuracy")
plt.legend(["train", "test"])
plt.grid()
plt.show()

#%% Loss

plt.plot(hist.history["loss"]) #default color: cyan
plt.plot(hist.history["val_loss"])
plt.title("Loss Graph")
plt.xlabel("epocs")
plt.ylabel("Loss")
plt.legend(["train", "test"])
plt.grid()
plt.show()
#%%
#CIFAR-10 에 대해서 실습해 볼 것 : 프로그램 7-6