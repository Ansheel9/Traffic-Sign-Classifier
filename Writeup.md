Project 3 - Traffic Sign Classifier :warning: 
===

Udacity Self-Driving Car Engineer Nanodegree

## Table of Contents

 - [ Summary ](#sum)
 - [ Description of Pipeline ](#des)
      - [ Data Set Summary & Exploration ](#cam)
      - [ Design and Test a Model Architecture ](#dis)
      - [ Test a Model on New Images ](#bin)
 - [ Potential Shortcomings ](#short)
 - [ Improvement to Pipeline ](#fut)

<a name="sum"></a>
## Summary

![](https://github.com/Ansheel9/Traffic-Sign-Classifier/blob/master/grayscale.jpg)

In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

<a name="des"></a>
## Description of Pipeline
---
My pipeline consisted of 3 major steps. First, I performed dataset summary & exploration. Then I defined a model architecture & trained on dataset of german traffic signs. Third, the model was tested on the five german traffic sign downloaded fron the web. Brief description of each step is given as follow:

<a name="cam"></a>
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The Dataset used is German Traffic Signs Dataset which contains images of the shape (32x32x3) i.e. RGB images.
I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Visualization of the dataset is done by a very simple basic approach is taken to display a single image from the dataset.

![](https://github.com/Ansheel9/Traffic-Sign-Classifier/blob/master/grayscale.jpg)

<a name="dis"></a>
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle my X_train, y_train. Then, I converted my RGB image to Grayscale & also, used Normalization as one of the preprocessing technique. In which, the dataset (X_train, X_test, X_valid) is fed into the normalization(x_label) function which converts all the data and returns the normalized one.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution Layer 1   	| Outputs 28x28x6 	|
| RELU					|	Activation applied to output of layer 1	|
| Pooling	      	| Input = 28x28x6, Output = 14x14x6 				|
| Convolution Layer 2	    | Outputs 10x10x16    									|
| RELU		| Activation applied to output of layer 2        									|
| Pooling				| Input = 10x10x16, Output = 5x5x16        									|
|	Flatten					|		Input = 5x5x16, Output =400										|
|		Fully Connected Layer 1				|		Input = 400, Output = 120						|
| RELU		| Activation applied to output of Fully Connected layer 1        									|
|		Fully Connected Layer 2				|		Input = 120, Output = 84						|
| RELU		| Activation applied to output of Fully Connected layer 2        									|
|		Fully Connected Layer 3				|		Input = 84, Output = 43					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used: 
EPOCHS = 30, 
BATCH_SIZE = 128, 
rate = 0.001, 
mu = 0, 
sigma = 0.1. 
I used the same LeNet model architecture which consists of two convolutional layers and three fully connected layers. The input is an image of size (32x32x1) and output is 43 i.e. the total number of distinct classes. In the middle, I used RELU activation function after each convolutional layer as well as the first two fully connected layers. Flatten is used to convert the output of 2nd convolutional layer after pooling i.e. 5x5x16 into 400. Pooling is also done in between after the 1st and the 2nd convolutional layer. The training of the model is calculated in cell 13 and the model architecture is defined in cell.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.937

I used normalized images to train the model and the number of EPOCHS=30 and the BATCH_SIZE=128. With the use of the defined hyperparameters, the validation set accuracy is 0.937 which is more than the 0.93 benchmark. EPOCHS and BATCH_SIZE are defined in cell and the normalization of the dataset is done in cell before. Moreover, the shuffling of the dataset is done. Further, the model has an accuracy of 0.6 on the five downloaded images of german traffic signs from the web.
 

<a name="bin"></a>
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

 Here are Five German traffic signs that I found on the web:

![test_1](https://github.com/Ansheel9/Traffic-Sign-Classifier/blob/master/data/traffic1.png)
![test_2](https://github.com/Ansheel9/Traffic-Sign-Classifier/blob/master/data/traffic2.png)
![test_3](https://github.com/Ansheel9/Traffic-Sign-Classifier/blob/master/data/traffic3.png)
![test_4](https://github.com/Ansheel9/Traffic-Sign-Classifier/blob/master/data/traffic4.png)
![test_5](https://github.com/Ansheel9/Traffic-Sign-Classifier/blob/master/data/traffic5.png)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction on the eight german traffic signs:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road    		| Priority road									| 
| General caution     			| General caution 										|
| Keep right				| Priority road												|
| General caution	      		| General caution					 				|
| Keep right			| Speed limit (60km/h) 							|

The model was able to correctly guess 3 of the 5 traffic signs. Test Accuracy on new loaded images = 0.6.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Ipython notebook and the code for evaluating the accuracy on these new downloaded images is displayed in the next cell. 

The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.13296957       			| Priority road	  									|

For the second image, the model is sure that this is a General caution sign, and the image do contain a General caution sign. Hence, predicted correctly. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.13137311        			| General caution   									| 

For the third image, the model is unsure that this is a correct sign (probability evenly distributed). Hence, not predicted correctly. The top five soft max probabilities were:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.16518918        			| Priority road	   									| 
| 0.12350544    				|  	Keep Right									|

For the fourth image, the model is sure that this is a Genral caution sign, and the image do contain a Genral caution sign. Hence, predicted correctly. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.18929006        			| Genral caution   									| 

For the fifth image, the model is unsure of the correct sign (probability was evenly distributed). Hence, not predicted correctly. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.16643417       			| Speed limit (60km/h)  									| 
| 0.11010567     				| Speed limit (80km/h)										|




<a name="short"></a>
## Potential Shortcomings with Current Pipeline
---
 - The prediction of the external web images were not accurate.
 - The accuracy of the model can be improved.

<a name="fut"></a>
## Improvement to Pipeline
---
 - Using Data Augmentation methods such as rotation, flip, etc., accuracy of the model can be improved.
 - Advanced architecture like ResNet & DeepLab can be utilized to improve accuracy.


