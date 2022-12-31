#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# # Loading - "MNIST Data Set"
# ## Containing Training samples = 60000,    Testing Samples = 10000
# ### Tensorflow already contain MNIST data set which can be loaded using Keras

# In[9]:


mnist = tf.keras.datasets.mnist ## this is basically handwritten characters based on 28x28 sized images of a 0 to 9


# # After loading the MNIST data, Divide into train and Test datasets  

# In[ ]:





# In[4]:


## unpacking the dataset into train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[6]:


x_train.shape


# In[8]:


import matplotlib.pyplot as plt 
plt.imshow(x_train[0])
plt.show() ## in order to execute the graph
## however we don't know whether its color image or binary images
## so inorder to plot it change the configuration
plt.imshow(x_train[0], cmap = plt.cm.binary)


# # Checking the values of each pixel
# # Before Normalization
# 

# In[10]:


print(x_train[0]) ## before normalization


# ## As images are in Gray level (1 channel ==> 0 to 255), not Colored(RGB)
# # Normalizing the data | Pre-Processing Step

# In[12]:


### you might have noticed that, its gray image and all values varies from 0 to 255
### in order to normalize it
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0], cmap = plt.cm.binary)


# # After Normalization

# In[13]:


print(x_train[0]) ## you can see all values are now normalized


# In[14]:


print(y_train[0]) ### just to check that we have labels inside our network


# ## Resizing image to make it suitable for apply convolution operation

# In[15]:


import numpy as np
IMG_SIZE=28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1) ### increasing  one dimension for kernel(or filter) operation
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1) ### increasing one dimension for kernel operation
print("Training Samples dimension", x_trainr.shape)
print("Testing samples dimension", x_testr.shape)


# # Creating a Deep Neural Network
# ### Training an 60000 samples of MNIST handwritten dataset

# In[18]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# In[19]:


#### creating a neural network now
model = Sequential()

#### First convolution layer 0 1 2 3 (60000,28,28,1)  28-3+1= 26x26
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) ### only for first convolution layer to mention input layer size
model.add(Activation("relu")) ## activation function to make it non-linear, <0, remove , > 0
model.add(MaxPooling2D(pool_size=(2,2))) ## MAXpooling  single maximum value of 2x2,

#### 2nd convolution layer  26-3+1=24*24
model.add(Conv2D(64, (3,3))) ## 2nd convolution layer
model.add(Activation("relu")) ## activation function 
model.add(MaxPooling2D(pool_size=(2,2))) ## MAXpooling

#### 3rd convolution layer
model.add(Conv2D(64, (3,3))) #  24*24
model.add(Activation("relu")) ## activation function 
model.add(MaxPooling2D(pool_size=(2,2))) 

#### Fully connected layer # 1  20x20 = 400
model.add(Flatten()) ### before using fully connected layer, need to be flatten so that 2D to 1D
model.add(Dense(64)) # 
model.add(Activation("relu"))

#### Fully connected layer #2
model.add(Dense(32))
model.add(Activation("relu"))

#### Last fully connected layer, output must be equal to number of classes, 10 (0-9)
model.add(Dense(10)) ## this last dense layer must be equal to 10
model.add(Activation('softmax')) ### activation function is changed to softmax (class probabilities)
## if we had a binary classification, one neuron in Dense Layer, sigmoid


# In[20]:


model.summary()


# In[21]:


print("Total Training Samples = ", len(x_trainr))


# In[22]:


model.compile(loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[23]:


model.fit(x_trainr, y_train, epochs=5, validation_split = 0.3) ## Training my model


# In[24]:


### Evaluating on testing data set MNIST
test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Test Loss on 10000 test sample", test_loss)
print("validation Accuracy on 10000 test samples", test_acc)


# In[25]:


# predictions = new_model.predict([x_test]) ## there is specialized method for efficiently saving your model, to name all input
### therefore instead of using new model loaded, for now only for predictions I am using simple model
predictions = model.predict([x_testr])


# In[26]:


print(predictions)


# In[27]:


print(np.argmax(predictions[0])) ### so actually argmax will return the maximum value index and find the value of it


# In[28]:


### now check that our answer is true or not
plt.imshow(x_test[0])


# In[29]:


## in order to underdand, convert the predictions from one hot encoding, we need to use numpy for that 
print(np.argmax([predictions[128]])) ### so actually argmax will return the maximum value index and find the value of it


# In[30]:


### now to check that is our answer is true or not
plt.imshow(x_test[128])


# In[31]:


import cv2 


# In[32]:


img = cv2.imread('digit1.png')


# In[35]:


plt.imshow(img)


# In[39]:


img.shape


# In[38]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[40]:


gray.shape


# In[41]:


resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)


# In[42]:


resized.shape


# In[43]:


newimg = tf.keras.utils.normalize(resized, axis = 1) ## 0 to 1 scaling


# In[44]:


newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # kernel operation of convolution layer


# In[45]:


newimg.shape


# In[46]:


predictions = model.predict(newimg)


# In[47]:


print(np.argmax(predictions))


# In[ ]:




