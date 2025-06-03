---
layout: post
title:  "Understanding Neural Networks"
date:   2025-03-28 16:35:07 +0100
categories: jekyll update
---
<link rel="stylesheet" href="/Assets/css/style.scss">
<link rel="stylesheet" href="{{ '/Assets/css/style.scss' | relative_url }}">
<span style="color:black">
# **Understanding neural networks**
The objective of this post is to gain an understanding of the concepts such as weights of a model and the operations involved in a simple neural network.
Classification using multinomial logistic regression is a good starting point.
Consider a scenario where you have a model that takes two inputs, the age of a person and the gender of a person and classifies them into three classes where each class represents the genre of movie that they most like.
<p align="center">
<img src='/Assets/Images/movie_class.png' class="center" />
</p>
Let us use a mathematical model called multinomial logistic regression to solve the task. I chose the multinomial logistic regression as it is a predeccesor to a fully connected neural network.
It is simply a neural network with a single layer. For our classification task, consider the following illustration. 

<p align="center">
<img src='/Assets/Images/MultinomialLogisticRegression.png' class="center" />
</p>

The illustration shows a perceptron and a multi-layer perceptron is a neural network. For a given class C, let us have a function to capture the relationship between movies of genre C and
the age and the gender of an adult.
f(Ci) = w0 + w1*age + w2*gender where f(C) determines how the age and gender can determine whether an adult prefer a movie that belongs to class C. The coefficients w1 and w2 are 
used to determine the factor by which age and gender determine that an adult likes a movie of class C.
f(C=thriller) = w00 + w01*age + w02*gender	(1)
f(C=Comedy) = w10 + w11*age + w12*gender	(2)
f(C=Action) = w20 + w21*age + w22*gender	(3)

Now taking only the coefficients of age and gender in the three functions we get a matrix of size 3*2,

<p align="left">
<img src='/Assets/Images/weight_matrix.png' class="center" />
</p>

Equations (1), (2) and (3) can be written as 
<p align="left">
<img src='/Assets/Images/MultinomialLogisticRegressionEqn.png' class="center" />
</p>
Now, we will have to map the outputs of each of the functions into probabilities i.e the probability that an adult with a given age and gender will prefer a particular genre of movie.
In order to do that we use the softmax function which is given by
e^zi/sum(e^zi) or in our case, the probability that an adult with a given age and gender will prefer comedies is
P(x|comedy) = e^f(C=comedy)/ e^f(C=thriller) + e^f(C=comedy) + e^f(C=action)
Similarly,
P(x|action) = e^f(C=action)/ e^f(C=thriller) + e^f(C=comedy) + e^f(C=action)
P(x|thriller) = e^f(C=thriller)/ e^f(C=thriller) + e^f(C=comedy) + e^f(C=action)
Note that the sum of all the probabilities equal to one.
Going back to equation (1), w is called the weight matrix and b is also called the bias vector. These values together are called the parameters of the model and are 'trained' or updated in
each iteration. It is the optimal value of these parameters that we want to find by the end of the training.
We then assign the class with the highest probability to the input. We now need to define a loss function that will determine if correctness of the predicted label.
A commonly used loss function is called the cross entropy loss.
Let us work this out with a numerical example.
<br/>
Consider the following data
<p align="left">
<img src='/Assets/Images/movie_dataset.png'/>
</p>

Let us encode 'Male' as 0 and Female as '1'. Likewise, Action=0, Thriller=1, Comedy=2.

Let the weight matrix and the bias vectors initially be

<p align="left">
<img src='/Assets/Images/weight_bias_values.png'/>
</p>

For the first input, <b>[23, 0]</b>

<p align="left">
<img src='/Assets/Images/iter1.png'/>
</p>
The probabilites that the user falls into each of the categories is calculated as follows.	
<p align="left">
<img src='/Assets/Images/probabilities.png'/>
</p>
The highest probability is assigned to P(Action) and therefore the output or the predicted label is that the user likes Action movies which in this case turns out to be true.
Now let us compute the loss function or the error which will later determine how the parameters of the model are to be updated. 
We will use the cross entropy loss to compute the loss. Before we do that, let us represent the output using one-hot encoding. 
<br/>
<b>
Action = [1 0 0]
<br/>
Thriller = [0 1 0]
<br/>
Comedy = [0 0 1]
<br/>
</b>
Cross entropy loss is given by
<br/>
<b>L = -sum( y(k) * log(pk) </b> where y(k) is the actual output or the label for the input for class k and pk is the proability of the inpput belonging to class k.
<br/>
So in the case of the input above, the actual label or the class for the input is 'Action' or [1 0 0].
Since the highest probability was assigned to the class='Action', the output is <b>[1 0 0]</b>.
<br/>
<b>
y[class=Action] = 1, p[class=Action] = 0.4378
<br/>
y[class=Thriller]=0, p[class=Thriller]=0.238
<br/>
y[class=Comedy]=0, p[class=comedy]=0.324
<br/>
</b>
Cross entropy loss is then calculated to be,
<b>L = -(1 * log(0.4378) + 0*log(0.238) + 0*log(0.324)) = 0.825</b>
<br/>
In order to compute the change in loss with respect to input, we need dL/dW.
The derivative of the loss function with respect to the weight matrix is given by
<br/>
dL/dW = (p - y).x
<br/>
The derivative of the loss function with respect to the bias is given by
<br/>
dL/db = (p-y)
<br/>
<b>p - y = [0.438−1, 0.238−0, 0.324−0]=[−0.562, 0.238, 0.324]</b>
<br/>
<p align="left">
<img src='/Assets/Images/gradient_descent.png'/>
</p>

Using the gradient descent algorithm to update the model parameters (weights and the bias vector) using a learning rate alpha = 0.1.
<br/>
W = W - alpha * dL/dW
<br/>
B = B - alpha * dL/dB
<br/>

<p align="left">
<img src='/Assets/Images/update_parameters.png'/>
</p>
Now, the new weight matrix and the bias vector will be used for the next input. Once the entire input dataset is trained, we have 1 epoch of the training completed. The training is repeated for multiple epochs till the network converges or
the loss value is nearly zero. 
 
**Inputs in batches**
<br/>
Consider a fully connected neural network that classifies images into one of the 10 classes. A classic example of such a dataset is the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) handwritten digit database. 
Each image is of size 28\*28 and each pixel represents either a 1 or 0 (Binary image).The size of each image is hence 28\*28=784. The dataset consists of a total 60,000 images. A fully connected layer will have 784 nodes in the input layer and 10 output nodes
or neurons in the output layer. 
Typically, each layer in the neural netowrk is represented by a weight matrix. It is the values of these matrices that we want to determine at the end of the training processes.
The dimension of the weight matrix for each layer is of size m*n where m is the number of nodes or neurons in the layer and n is the number of nodes in the previous layer.
Let us assume that the second layer has 128 neurons. The size of the weight matrix *W* is then 128\*784 as the input layer which is the first layer accepts 784 inputs (for the classification of the MNIST dataset).
<br/>
Now usually, for performance reasons, one does not train a network using thousands of images, image by image.
That leads to  a severe under utilization of the computing hardware. So, the input dataset is fed into the network in terms of batches. Let us assume that each batch is of size 64. In this
case, the input to the network is actually a matrix of 64\*784. In order to account for different batch sizes, most deep network framweworks create a input layer of size (batch size * size of input).
Now in each layer, we have the input multiplied with weight matrix of the layer.
<br/>
z = x * Transpose(W) + b. Size of x is 64\*784. Transpose(W) is of size 784\*128. Hence the output of the second layer is of size 64\*128. 
The loss values is then computed as the average of the losses due to each image in the batch. Note that the network or the weight matrix *W* does not have to change in order to accomodate the inputs in batches