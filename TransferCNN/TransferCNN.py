
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
import numpy as np
from keras import layers
from keras import models
from keras.applications import InceptionResNetV2
from sklearn.metrics import classification_report, confusion_matrix


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

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=25)#, featurewise_center=True)

batchSize = 64
datagenTraining = datagen.flow_from_directory('dataset/training_set', 
                            class_mode='binary', target_size=(150,150), batch_size=batchSize)
datagenTest = datagen.flow_from_directory('dataset/test_set', 
                            class_mode='binary', target_size=(150,150), batch_size=batchSize)

#datagen.fit()
model.compile(optimizer=keras.optimizers.Adam(0.01), 
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
model.summary()

loss = model.evaluate(datagenTest, steps=1)


model.fit(datagenTraining, epochs=1, steps_per_epoch=2000/batchSize)


#loss = model.evaluate(datagenTest, steps=1)


#Confution Matrix and Classification Report
datagenTest = datagen.flow_from_directory('dataset/test_set', 
                            class_mode='binary', target_size=(150,150), 
                            batch_size=batchSize, shuffle=False)
Y_pred = model.predict_generator(datagenTest, 2000 // batchSize+1)
y_pred = np.where(Y_pred > 0.5, 1, 0)
print('Confusion Matrix')
print(confusion_matrix(datagenTest.classes, y_pred))
print('Classification Report')
target_names = ['Cats', 'Dogs']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

a=5


