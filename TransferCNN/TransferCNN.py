
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
import numpy as np
from keras import layers
from keras import models
from matplotlib import pyplot as plt
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
flayer = preModel.get_layer(index=1)
filters = flayer.get_weights()[0]
print(flayer.name, filters.shape)

# Normalize filters
fmin, fmax = filters.min(), filters.max()
filters = (filters - fmin) / (fmax - fmin)

fig, axs = plt.subplots(4, 8)
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(filters[:,:,:,i*8+j])


plt.show()


def addTransferHead(otherModel):
    # Add transfer head to sequential model
    model = models.Sequential()
    model.add(otherModel)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


model = addTransferHead(preModel)
preModel.summary()
model.summary()

# Freeze weights
preModel.trainable=False

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=25)#, featurewise_center=True)
datagenTest = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
datagenTestIO = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255) # Weird bug requires duplicating datagen's if used twice

batchSize = 64
datagenTraining = datagen.flow_from_directory('dataset/training_set', 
                            class_mode='binary', target_size=(150,150), batch_size=batchSize)

datagenTest = datagenTest.flow_from_directory('dataset/test_set', 
                            class_mode='binary', target_size=(150,150), batch_size=batchSize)

datagenTestIO = datagenTestIO.flow_from_directory('dataset/test_set', 
                            class_mode='binary', target_size=(150,150), 
                            batch_size=batchSize, shuffle=False)



newModel = models.Sequential()
for layer in preModel.layers[0:17]:
    newModel.add(layer)
newModel.trainable = False
newModel = addTransferHead(newModel)

def printConfusionMatrix(model, datagen):
    #Confusion Matrix and Classification Report
    Y_pred = model.predict(datagenTestIO, 2000 // batchSize+1)
    y_pred = np.where(Y_pred > 0.5, 1, 0)
    print('Confusion Matrix')
    print(confusion_matrix(datagenTestIO.classes, y_pred))
    print('Classification Report')
    target_names = ['Cats', 'Dogs']
    print(classification_report(datagenTestIO.classes, y_pred, target_names=target_names))


def plotAccuracy(name, history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(name + '_acc.jpg')
    plt.close()

def plotLoss(name, history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(name + '_loss.jpg')
    plt.close()

def trainModel(model, epochs, name, lr=0.002):
    model.compile(optimizer=keras.optimizers.Adam(lr), 
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
    printConfusionMatrix(model, datagen)
    history = model.fit(datagenTraining, epochs=epochs, #steps_per_epoch=1000//batchSize, # 5 epochs seems best so far
              validation_data=datagenTest)#, validation_steps=500//batchSize)
    printConfusionMatrix(model, datagen)
    plotAccuracy(name, history)
    plotLoss(name, history)
    


trainModel(model, 12, 'transferHead')
#trainModel(newModel, 20, 'reducedTransfer', lr=0.01)










