
import keras
from keras.applications import InceptionResNetV2

preModel = InceptionResNetV2(weights='imagenet', 
                             include_top=False, input_shape=(150,150,3))

