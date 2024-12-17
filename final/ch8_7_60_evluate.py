import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

#%% Data Loading

data_path=pathlib.Path("datasets/Stanford_dogs/images") #image file folder directory

train_ds=image_dataset_from_directory(data_path, validation_split=0.2, subset="training",
                                       seed=123, image_size=(224, 224), batch_size=16)
test_ds=image_dataset_from_directory(data_path, validation_split=0.2, subset="validation",
                                       seed=123, image_size=(224, 224), batch_size=16)
#save folder name to class_name as class name

class_names = train_ds.class_names
print(class_names)
#%% Model Construction

#base_model=DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
cnn=tf.keras.models.load_model('cnn_for_stanford_dogs_60.h5')	# 모델 읽기
#%% compile and fit


res=cnn.evaluate(test_ds, verbose=2)
print("Accuracy", res[1]*100)
print("val-loss", res[0]*100)

