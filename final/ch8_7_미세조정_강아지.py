# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:57:57 2023

@author: BigData
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

#%% Data Loading

data_path=pathlib.Path('datasets/Stanford_dogs/images') #image file folder directory

train_ds=image_dataset_from_directory(data_path, validation_split=0.2, subset="training",
                                       seed=123, image_size=(224, 224), batch_size=16)
test_ds=image_dataset_from_directory(data_path, validation_split=0.2, subset="validation",
                                       seed=123, image_size=(224, 224), batch_size=16)
#save folder name to class_name as class name

class_names = train_ds.class_names
print(class_names)
#%% Model Construction

base_model=DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
cnn=Sequential()
cnn.add(Rescaling(1./255.))
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024, activation="relu"))
cnn.add(Dropout(0.75))
cnn.add(Dense(units=120, activation="softmax"))

#%% compile and fit

cnn.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.000001), metrics=["accuracy"])
hist=cnn.fit(train_ds, epochs=200, validation_data=test_ds, verbose=2)

print("Accuracy", cnn.evalute(test_ds, verbose=0)[1]*100)

cnn.save("cnn_for_stanford_dogs.h5")

#%% save class name  to file
import pickle
f=open("dog_species_names.txt", "wb")
pickle.dump(train_ds.class_names, f)
f.close()

#%% Dwraw Accuracy and Loss

import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"]) #default color: cyan
plt.plot(hist.history["val_accuracy"])
plt.title("Accuracy Graph")
plt.xlabel("epocs")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"])
plt.grid()
plt.show()

#%% Loss

plt.plot(hist.history["loss"]) #default color: cyan
plt.plot(hist.history["val_loss"])
plt.title("Loss Graph")
plt.xlabel("epocs")
plt.ylabel("Loss")
plt.legend(["train", "validation"])
plt.grid()
plt.show()