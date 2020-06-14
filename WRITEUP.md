# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./steering_hist.png "Steering histogramm"
[image2]: ./steering_hist_after.png "Steering histogramm after removing 70 % of zero steerings"
[image3]: ./network_arch.png "Network architecture"
[video1]: ./video.mp4 "Video of the car driving autonomously around the track"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* WRITEUP.md to summarizing the results (this file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on a convolution neural network developed by NVIDIA ([link to the paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)) just like proposed oin the lessons (`model.py` lines 114-134).

The model includes ReLU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 117). Range -0.5 to +0.5 as proposed in the lessons seems to be very narrow, so I increased the range to +-1.

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 127, 129). I  also incorporated early stopping (3 epochs) to counter overfitting.

The model was trained and validated on different data sets (split 90 % : 10 %, before augmentation) to ensure that the model was not overfitting (code line 39). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. Additional tuning of learning rate was not neccessary, the base learning rate of 0.001 was already doing great (code line 148).

#### 4. Appropriate training data

I evaluated the provided training data. It appears that the training data was recorded using a keyboard as it is dominated by zero steerings (see histogramm below). 

![Steering histogramm][image1]

To remove the bias of zero steerings and more importantly, to remove the bias of +- offset steerings when using the left and right picture, I decided to remove 70 % of the zero steering training data (see histogramm below). 

![Steering histogramm after removing 70 % of zero steerings][image2]

This also helped to reduce the training time. The provided training data seems to have enough center lane driving and recovering from the left and right sides of the road as the model runs smoothly without the need of additional data. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

I started out with analog to the route shown in the lessons. After splitting the provided labeled image set into a training and validation set, I trained a small network (LeNet) using only the center images (cropped and normalized using a lambda layer) with their corresponding steering angles and used the saved `model.h5` to run the simulator in autonomous mode. That way, I was able to test the pipeline. Surprisingly, the model did already quite well and managed to get around the first corner. 

As proposed in the lessons, I replaced the small LeNet with a more sophisticated network from NVIDIA ([link to the paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)). Subsequently, I added augmentation techniques such as flipping and using the left and right camera with an offset to the target steering angle of 0.20. All of the augmentation added 6 times as much data, so I added a generator function and load the data in augmented batches (lines 66-107). 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Finally, I got to improve the driving behavior when I switched from BGR (standard setting in cv2) to RGB images in the training period, as the `drive.py` is also using RGB images. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 

![Video of the car driving autonomously around the track][video1]

#### 2. Final Model Architecture

The final model architecture (model.py lines 114-134) consisted of a convolution neural network with the following layers and output layer shapes. 

```console
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 85, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 41, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 19, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```

Here is a visualization of the architecture:

![Network architecture][image3]

#### 3. Creation of the Training Set & Training Process

I only used the provided training data with the augmentation techniques shown above. In the generator, the data is randomly shuffled.  

The training and validation split helped determine if the model was over or under fitting. The ideal number of epochs was 3. While the mse value did not increase for the validation set, the subjective drining was worse for more epochs. MSE may not be the ideal metric here. 
