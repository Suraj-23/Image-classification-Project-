# -- coding: utf-8 --
#%matplotlib inline

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from glob import glob
import streamlit as st

st.set_page_config(layout="wide")



IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'


# In[3]:


train_path


# In[4]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[5]:


resnet.summary()


# In[6]:


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


# In[7]:


# useful for getting number of output classes
folders = glob('Datasets/train/*')


# In[8]:


folders


# In[9]:


len(folders)


# In[10]:


# our layers - you can add more if you want
x = Flatten()(resnet.output)


# In[11]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[12]:


# view the structure of the model
model.summary()


# In[13]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[14]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[15]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,class_mode = 'categorical')


# In[16]:


test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[17]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[18]:


r.history


# In[19]:


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


# In[20]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_resnet50.h5')


# In[21]:


y_pred = model.predict(test_set)


# In[22]:


y_pred


# In[23]:


import numpy as np 
y_pred = np.argmax(y_pred, axis=1)


# In[24]:


y_pred


# In[25]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[26]:


model=load_model('model_resnet50.h5')


# In[39]:


# img_data


# In[30]:


img=image.load_img('Datasets/Test/lamborghini/11.jpg',target_size=(224,224))


# In[31]:


x=image.img_to_array(img)
x


# In[32]:


x.shape


# In[33]:


x=x/255


# In[34]:


x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[35]:


model.predict(img_data)


# In[36]:


a=np.argmax(model.predict(img_data), axis=1)


# In[37]:


a


# In[38]:


a==1


# In[ ]:




