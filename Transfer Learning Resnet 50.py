#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning VGG 16 and VGG 19 using Keras
# 

# In[1]:


get_ipython().system('pip install streamlit')


# In[2]:


# import the libraries as shown below
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from glob import glob
import streamlit as st


# In[3]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'


# In[4]:


train_path


# In[5]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[6]:


resnet.summary()


# In[7]:


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


# In[8]:


# useful for getting number of output classes
folders = glob('Datasets/train/*')


# In[9]:


folders 


# In[10]:


len(folders)


# In[11]:


# our layers - you can add more if you want
x = Flatten()(resnet.output)


# In[12]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[13]:


# view the structure of the model
model.summary()


# In[14]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[15]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[16]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,class_mode = 'categorical')


# In[17]:


test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[18]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[19]:


r.history


# In[20]:


# plot the loss
plt.figure(figsize=(10,5))
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.figure(figsize=(10,5))
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[21]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_resnet50.h5')


# In[22]:


y_pred = model.predict(test_set)


# In[23]:


y_pred


# In[24]:


import numpy as np 
y_pred = np.argmax(y_pred, axis=1)


# In[25]:


y_pred


# In[26]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[27]:


model=load_model('model_resnet50.h5')


# In[28]:


# img_data


# In[30]:


img=image.load_img('Datasets/Test/lamborghini/11.jpg',target_size=(224,224))


# In[31]:


image


# In[32]:


x=image.img_to_array(img)
x


# In[33]:


x.shape


# In[34]:


x=x/255
x


# In[35]:


x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[36]:


model.predict(img_data)


# In[37]:


a=np.argmax(model.predict(img_data), axis=1)


# In[38]:


a


# In[39]:


a==1

