#C-C-P-D structure

import tensorflow as tf
import tensorflow.keras.datasets as ds
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
#%% Data handling
(x_train, y_train), (x_test, y_test)=ds.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train=x_train.astype(np.float32)/255.0  #0~1 normalization
x_test=x_test.astype(np.float32)/255.0 
y_train=tf.keras.utils.to_categorical(y_train, 10)
y_test=tf.keras.utils.to_categorical(y_test, 10)

#%% CNN Modeling
cnn=Sequential()
cnn.add(Conv2D(32, (3,3), activation="relu", input_shape=(32, 32,3)))
cnn.add(Conv2D(32, (3,3), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,2)))
#드롭아웃 층: 학습 후 네트워크의 각 노드들이 서로 비슷한 역할을 하는 것 회피
#랜덤하게 연속된 몇 개의 node를 지운다.(DropBlock)
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64, (3,3), padding='valid',activation='relu'))
cnn.add(Conv2D(64, (3,3), padding='valid',activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(units=512, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=10, activation="softmax"))

cnn.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
hist=cnn.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), verbose=2)

res=cnn.evaluate(x_test, y_test, verbose=0)
print("Accuracy", res[1]*100)

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