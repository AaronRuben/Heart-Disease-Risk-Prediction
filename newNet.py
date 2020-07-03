#!/usr/bin/env python
# coding: utf-8

# In[1]:


# first neural network with keras tutorial

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from keras.optimizers import SGD


# In[2]:


# load the dataset
df_empty = np.genfromtxt('framingham.csv', delimiter=",")
print(df_empty.shape)
print(df_empty)


# In[3]:


imputer = KNNImputer(n_neighbors=3, weights="uniform")
df = imputer.fit_transform(df_empty)


# In[4]:


# split into input (X) and output (y) variables
X = df[:,0:15]
y = df[:,15]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[5]:


# define the keras model
model = Sequential()
model.add(Dense(25, input_dim=15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[6]:


# compile the keras model

opt = SGD(lr=0.0001)


model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[7]:


# Fit the model
history = model.fit(X_train, y_train, validation_split=0.33, epochs=150, batch_size=10)


# In[8]:


history_dict = history.history
print(history_dict.keys())


# In[9]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[10]:


y_pred = model.predict_classes(X_test)
print(accuracy_score(y_test, y_pred))


# In[ ]:




