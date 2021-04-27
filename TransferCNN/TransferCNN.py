
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
import numpy as np
from keras import layers
from keras import models
from keras.applications import InceptionResNetV2
from sklearn.metrics import classification_report, confusion_matrix

#import tensorflow as tf
#config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
##device_count = {'GPU': 1}
#)
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)


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

batchSize = 32
datagenTraining = datagen.flow_from_directory('dataset/training_set', 
                            class_mode='binary', target_size=(150,150), batch_size=batchSize)
datagenTest = datagen.flow_from_directory('dataset/test_set', 
                            class_mode='binary', target_size=(150,150), batch_size=batchSize)
datagenTestIO = datagen.flow_from_directory('dataset/test_set', 
                            class_mode='binary', target_size=(150,150), 
                            batch_size=batchSize, shuffle=False)

#datagen.fit()
model.compile(optimizer=keras.optimizers.Adam(0.01), 
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
model.summary()

def printConfusionMatrix(model, datagen):
    #Confusion Matrix and Classification Report
    Y_pred = model.predict(datagenTestIO, 2000 // batchSize+1)
    y_pred = np.where(Y_pred > 0.5, 1, 0)
    print('Confusion Matrix')
    print(confusion_matrix(datagenTestIO.classes, y_pred))
    print('Classification Report')
    target_names = ['Cats', 'Dogs']
    print(classification_report(datagenTestIO.classes, y_pred, target_names=target_names))

printConfusionMatrix(model, datagen)
model.fit(datagenTraining, epochs=5, #steps_per_epoch=500//batchSize, # 5 epochs seems best so far
          validation_data=datagenTest, validation_steps=500//batchSize)
printConfusionMatrix(model, datagen)







