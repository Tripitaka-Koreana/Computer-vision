import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
(x_train, y_train), (x_test, y_test)=ds.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train=x_train.astype("float32")/255.0  #0~1 normalization
x_train=x_train[0:15]; y_train=y_train[0:15]

class_names=["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

plt.figure(figsize=(20,2))
plt.suptitle("First 15 images in train set")

for i in range(15):
    plt.subplot(1,15,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.xticks([])
    plt.title(class_names[int(y_train[i])])
    
plt.show()

#%% Data Generation

batch_siz=4
#데이터 증강: 데이터셋의 수를 늘리기 위해 같은 데이터를 조금씩 변형(회전,이동,명암 등등)하여 새로운 데이터 생성
generator=ImageDataGenerator(rotation_range=20.0, width_shift_range=0.2,
                             height_shift_range=0.2, horizontal_flip=True)
gen=generator.flow(x_train, y_train, batch_size=batch_siz)

for a in range(3):
    img, label=next(gen)
    plt.figure(figsize=(8, 2.4))
    plt.suptitle("Generator trial" + str(a+1))
    for i in range(batch_siz):
        plt.subplot(1, batch_siz, i+1)
        plt.imshow(img[i])
        plt.xticks([]); plt.xticks([])
        #print("Label", i, label)
        plt.title(class_names[int(label[i])])
    plt.show()


