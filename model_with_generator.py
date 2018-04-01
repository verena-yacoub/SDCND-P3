# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 01:29:50 2018

@author: Verena Maged
"""

import os
import csv
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D 
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn




lines =[] # generate an empty list for the raw data from the csv file
with open ('data/driving_log1.csv') as csvfile:
    reader= csv.reader(csvfile) # opening csv with csv reader
    for line in reader:
        lines.append(line) # append each line from csv into the created list


train_samples, validation_samples = train_test_split(lines, test_size=0.2)# use sklearn to split the data


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        images=[] # generate empty list to contain images
        measurments=[] # generate empty list to contain corresponding stirring measurment
        
        for line in lines:
            for i in range(3): # as in each line the first 3 cells contain images directories
                souce_path= str (line[i]) # assign the directory to a string
                filename=souce_path.split('/')[-1] # separate the string when the symbol "/" is found and take the last substring having the filename
                current_path= ('data/IMG/')+filename # add the filename to the correct root directory
                image= cv2.imread(current_path) # read the image from the constructed path
                images.append(image) #append the image read to the empty list images 
                measurment=float(line[3]) # set the measurment corresponding to the image as the 4th cell in the csv file
                if i==1: # cell containing the directory for the left camera image in the csv file
                    measurment+=0.2 # augmenting the clockwise stirring reaction for all left images
                if i==2: # cell containing the directory for the right camera image in the csv file
                    measurment-=0.2 # augmenting the anti clockwise stirring reaction for all right images
                measurments.append(measurment) # append the correct measurment to measurments list 
            
            
        augmented_images, augmented_measurments= [],[] # prepare empty list to contain original data and new data
        for image, measurment in zip(images, measurments): 
            augmented_images.append(image) #Append the original image to the data 
            augmented_measurments.append(measurment) # append the corresponding measurments
            augmented_images.append(cv2.flip(image,1)) # Flip the original image horizontally and append the new flipped to the dataset
            augmented_measurments.append(measurment*-1.0) # add the measurment to the list with a changed sign as the direction of the image was flipped
            augmented_images.append(cv2.threshold(image, 200,255,cv2.THRESH_BINARY )[1]) # append binary thresholded image to the images 
            augmented_measurments.append(measurment) # append corresponding measurments
            augmented_images.append(image+15) # append image with brightness enhanced 
            augmented_measurments.append(measurment) # append the corresponding measurment
            
        X_train= np.array(augmented_images) #define the training images as array
        y_train=np.array(augmented_measurments) #define the corresponding measurments as array     
        yield sklearn.utils.shuffle(X_train, y_train)
     
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model= Sequential() # initialize sequential model
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3))) # normalize the images and define inputs shape
model.add(Cropping2D(cropping=((70,25),(0,0)))) # cropping images to remove undesired parts
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu')) # add 5x5 convolution layer with stride 2 and depth of 24
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu')) # add 5x5 convolution layer with stride 2 and depth of 36
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu')) # add 5x5 convolution layer with stride 2 and depth of 48
model.add(Convolution2D(64,3,3, activation='relu')) # add 3x3 convolution layer with depth of 64
model.add(Convolution2D(64,3,3, activation='relu')) # add 3x3 convolution layer with depth of 64
model.add(Flatten()) # flatten the output in one vector
model.add(Dense(100)) # add a fully connected layer with 100 outputs
model.add(Dense(50)) # add a fully connected layer with 50 outputs
model.add(Dense(10)) # add a fully connected layer with 10 outputs
model.add(Dense(1)) # add a fully connected layer with 1 output


model.compile(loss='mse', optimizer='adam') # compile the model while calculating the loss with mean square method and tuning hyperparameters using adam optimizer 
early_stop = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=0,verbose=0,mode='auto') # In this callback validation loss is monitored such that if it is worsening by more than 0.0001 epochs are stopped 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 0, callbacks=[early_stop]) # fit the model and split the data to 20% validation and 80% training
model.save('model.h5') # saving the model to file

    
