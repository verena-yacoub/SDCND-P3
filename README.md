# SDCND-P3
# **Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"


## Rubric Points Discussion 
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **README.md** or summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The structure of the neural network was adopted from [NVIDIA self driving car model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) and was implemented as follows, [as per the code](https://github.com/verena-yacoub/SDCND-P3/blob/master/model.py#L69-L80):

* normalization layer using lambda function
* cropping layer to remove irrelevant parts like sky and trees 
* 5x5 convolution layer with stride 2 and depth of 24
* 5x5 convolution layer with stride 2 and depth of 36
* 5x5 convolution layer with stride 2 and depth of 48
* 3x3 convolution layer with depth of 64
* 3x3 convolution layer with depth of 64
* flatten the output 
* fully connected layer with 100 outputs
* a fully connected layer with 50 outputs
* a fully connected layer with 10 outputs
* a fully connected layer with 1 output

#### 2. Attempts to reduce overfitting in the model

* The main strategy to avoid overfitting was [data augmentation](https://github.com/verena-yacoub/SDCND-P3/blob/master/model.py#L49-L58) by flipping, thresholding, and brightness enhancement.
* The data was then [split](https://github.com/verena-yacoub/SDCND-P3/blob/master/model.py#L85) into 80% training and 20% validation 
* The model was examined in track one of the simulator and did not get off road 

#### 3. Model parameter tuning

The model used an [adam optimizer](https://github.com/verena-yacoub/SDCND-P3/blob/master/model.py#L83)

#### 4. Appropriate training data

* Training data was created by taking almost 2 laps around track one, while keeping the car centered as much as possible and occasionally recovering from left/right drifts for the model to learn handling different cases.
* The aforementioned data augmentation process is shown in the figure below.
![alt text][image1]


### Notes and discussion:
1- While loading stirring measurment angles, readings were emphased by a factor of 0.2 for the right and left images which resulted in a better response of the model at recovering from drifts 
*Diving deeper into this...:* we can find that +0.2 was [added](https://github.com/verena-yacoub/SDCND-P3/blob/master/model.py#L43) with the left side photos and -0.2 was [substracted](https://github.com/verena-yacoub/SDCND-P3/blob/master/model.py#L43) with the right side photos

