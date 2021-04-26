
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
from keras import layers
from keras import models
from keras.applications import InceptionResNetV2


preModel = InceptionResNetV2(weights='imagenet', 
                             include_top=False, input_shape=(150,150,3))

# Add transfer head to sequential model
model = models.Sequential()
model.add(preModel)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

preModel.summary()
model.summary()

# Freeze weights
preModel.trainable=False

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=25)



datagenTraining = datagen.flow_from_directory('dataset/training_set', 
                            class_mode='binary', target_size=(150,150), batch_size=32)
datagenTest = datagen.flow_from_directory('dataset/test_set', 
                            class_mode='binary', target_size=(150,150), batch_size=32)

model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.binary_crossentropy)
model.summary()

vals = model.evaluate()

a=5


