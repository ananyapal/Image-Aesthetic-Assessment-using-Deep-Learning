#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')

#%%
# Saving and loading model and weights
from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

num_channel=1

# Testing a new image
def testing(filename):
    test_image = cv2.imread(filename,1)
    test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image=cv2.resize(test_image,(128,128))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    print (test_image.shape)
       
    if num_channel==1:
    	if K.image_dim_ordering()=='th':
    		test_image= np.expand_dims(test_image, axis=0)
    		test_image= np.expand_dims(test_image, axis=0)
    	#	print (test_image.shape)
    	else:
    		test_image= np.expand_dims(test_image, axis=3) 
    		test_image= np.expand_dims(test_image, axis=0)
    	#	print (test_image.shape)
    		
    else:
    	if K.image_dim_ordering()=='th':
    		test_image=np.rollaxis(test_image,2,0)
    		test_image= np.expand_dims(test_image, axis=0)
    	#	print (test_image.shape)
    	else:
    		test_image= np.expand_dims(test_image, axis=0)
    	#	print (test_image.shape)
    		
    # Predicting the test image
    #print((model.predict(test_image)))
    print("Class is:")
    test_class=loaded_model.predict_classes(test_image)
    if(test_class==0):
        return "Appealing"
    else:
        return "Not Appealing"

path = '/Users/anuja/Desktop/Pro/TESTT/'

import os
listing  = os.listdir(path)
num_samples= len(listing)
print(num_samples)
count=0
num=0 
for i in range(num_samples):  
    
    img_name = str(i+1) + ".jpg"
    img_path = path + "//" + img_name
    print(img_path)
    exists = os.path.isfile(img_path)
    if exists:
        a=testing(img_path)
    if a == "Appealing":
        #count=count+1
        print("Appealing")
    else:
        #num=num+1
        print("not appealing")
#if count > num:
 #   print("Not Appealing")
#else:
 #   print("Appealing")
