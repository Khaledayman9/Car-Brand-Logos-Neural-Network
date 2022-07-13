from xml.etree.ElementInclude import include
import numpy as np 
import matplotlib as pl
import tensorflow as tf
from skimage import color
from tensorflow import keras
from keras.layers import Input,Dense,Activation,Dropout,Flatten,BatchNormalization,MaxPooling2D,Conv2D,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras_preprocessing import image
from keras.applications import vgg19
from keras.applications import vgg16
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser
from numpy import newaxis
import cv2
from glob import glob
from skimage.feature import local_binary_pattern
from keras.optimizers import Adam
from keras import optimizers
from keras import applications
from keras.regularizers import l2
from matplotlib.pyplot import figure
from PIL import Image

#../input/car-brand-logos/Car_Brand_Logos/Train/hyundai/images141.jpg"
hyundai_logo = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/hyundai/images141.jpg")
hyundai_logo = cv2.cvtColor(hyundai_logo, cv2.COLOR_BGR2RGB)
lexus_logo  = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/lexus/images103.jpg")
lexus_logo = cv2.cvtColor(lexus_logo, cv2.COLOR_BGR2RGB)
mazda_logo = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/mazda/0002.jpg")
mazda_logo = cv2.cvtColor(mazda_logo, cv2.COLOR_BGR2RGB)
mercedes_logo = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/mercedes/images12.jpg")
mercedes_logo = cv2.cvtColor(mercedes_logo, cv2.COLOR_BGR2RGB)
opel_logo = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/opel/00000aaa1.jpg")
opel_logo= cv2.cvtColor(opel_logo, cv2.COLOR_BGR2RGB)
skoda_logo = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/skoda/images156.jpg")
skoda_logo= cv2.cvtColor(skoda_logo, cv2.COLOR_BGR2RGB)
toyota_logo = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/toyota/images153.jpg")
toyota_logo= cv2.cvtColor(toyota_logo, cv2.COLOR_BGR2RGB)
volkswagen_logo = cv2.imread("../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/volkswagen/images137.jpg")
volkswagen_logo = cv2.cvtColor(volkswagen_logo, cv2.COLOR_BGR2RGB)


titles = ["hyundai","lexus","mazda","mercedes","opel","skoda","toyota","volkswagen"]
images = [hyundai_logo,lexus_logo,mazda_logo,mercedes_logo,opel_logo,skoda_logo,toyota_logo,volkswagen_logo]

figure(figsize=(18,16),dpi= 70)
for i in range(8):
    plt.subplot(4,2,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()    

print('shape:', volkswagen_logo.shape)
print('height:', volkswagen_logo.shape[0])
print('width: ', volkswagen_logo.shape[1])


def AverageFunct(list):
    return sum(list)/len(list)

path = "../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/"    
listdir = os.listdir(path)
ratios_sub = []
ratios = []

for imdir in listdir:
    imlist = os.listdir(os.path.join(path,imdir))
    for im in imlist:
        im = cv2.imread(os.path.join(path,imdir,im))
        ratio = (im.shape[0]/im.shape[1])
        ratios_sub.append((ratio))
    ratios_sub_average = AverageFunct(ratios_sub)
    ratios.append((ratios_sub_average))

ratios_average = AverageFunct(ratios)
print("The average ratio of width to heigth is {}".format(ratios_average))
width = 240
height = int(width*ratios_average)
input_shape = (height,width)
print(input_shape)
img_width,img_height=width,height 
training_data_dir = "../input/Car_Logos_Neural_Net/Car_Brand_Logos/Train/"    
testing_data_dir = "../input/Car_Logos_Neural_Net/Car_Brand_Logos/Test/"
number_of_classes = 8
batch_size = 8
epochs = 12

training_data_generator = ImageDataGenerator(rescale=1. /255,shear_range=0.95,zoom_range=0.95,horizontal_flip=True,vertical_flip=True)
testing_data_generator = ImageDataGenerator(rescale=1. /255)

train_generator = training_data_generator.flow_from_directory(
    training_data_dir,target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical')

test_generator = testing_data_generator.flow_from_directory(
    testing_data_dir,target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical')


vgg = vgg19.VGG19(include_top= False, weights = "imagenet", input_shape=( img_height,img_width, 3))
vgg_layer_list = vgg.layers

model = Sequential()

for layer in vgg_layer_list:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False
#Input_layer    
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
    
#Hidden_layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#Output_layer
model.add(Dense(number_of_classes))
model.add(Activation('softmax'))
model.summary()

opt = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999)

model.compile(optimizer= opt,loss = "categorical_crossentropy",metrics=["accuracy"])
hist = model.fit_generator(generator= train_generator,epochs= epochs,validation_data=test_generator)

print(hist.history.keys())
plt.plot(hist.history["loss"],label= "Train Loss")
plt.plot(hist.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"],label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"],label = "Validation Accuracy")
plt.legend()
plt.show()




# img_width,img_height = 244,244
# training_data_dir = 'v_Car_Brand_Logos\Train'
# testing_data_dir  = 'v_Car_Brand_Logos\Test'
# train_samples = 2513
# test_samples= 400
# carlogo_epochs = 10
# carlogo_batch_size = 16

# if K.image_data_format() == 'channels_first':
#     input_shape = (3,img_width,img_height)
# else:
#     input_shape = (img_width,img_height,3)
    
# model = Sequential()
# model.add(Conv2D(32,(2,2),input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(32, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss = 'binary_crossentropy',optimizer='rmsprop',metrices= ['accurary'])
# training_data_generator = ImageDataGenerator(rescale= 1. /255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
# testing_data_generator = ImageDataGenerator(rescale= 1. /255)

# training_generator = training_data_generator.flow_from_directory(training_data_dir,target_size=(img_width,img_height),batch_size=carlogo_batch_size, class_mode='binary')
# testing_generator = testing_data_generator.flow_from_directory(testing_data_dir,target_size=(img_width,img_height),batch_size=carlogo_batch_size ,class_mode='binary')

# model.fit_generator(training_generator,steps_per_epoch=train_samples,epochs =carlogo_epochs ,validation_data= testing_generator,validation_steps=test_samples)
