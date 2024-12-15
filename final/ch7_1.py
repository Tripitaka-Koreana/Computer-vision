import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

#%%

#dataset 확인하기
(x_train, y_train), (x_test, y_test)=ds.mnist.load_data() #MNIST dataset: 필기 숫자 데이터셋
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
plt.figure(figsize=(24,3))
plt.suptitle("MNIST", fontsize=30)

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.xticks([]); plt.xticks([])
    plt.title(str(y_train[i]), fontsize=30)
    
#%%
(x_train, y_train), (x_test, y_test)=ds.cifar10.load_data() #CIFAR10 dataset: 자연 영상 데이터셋
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
class_names=["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
plt.figure(figsize=(24,3))
plt.suptitle("CIFAR-10", fontsize=30)

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.xticks([])
    plt.title(class_names[y_train[i,0]], fontsize=30)
    
