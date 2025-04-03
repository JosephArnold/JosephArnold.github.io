---
layout: post
title:  "Understanding Backpropogation!"
date:   2024-11-18 16:35:07 +0100
categories: jekyll update
---

# Backpropogation
Backpropgation is a very efficient method of automatic differentiation that is used in neural networks where the gradient of the function to be optimized is to be computed.
Before we go into the details of the backpropogation, let us quickly recap the need for a automatic differentiation technique.
Consider a multivariate function,
f(x,y) that maps to a vector [u,v] where u = sin(4xy) and v = cos(2x + 2y)
Remember that the goal of a neural network is to optimize a function or to find the parameters that will result in the lowest value of the function or the global minima.
In order to find the global minima, we use the gradient of the function. The gradient of the function at a given point will always point to the steepest slope of the function in the ascending direction.
Now, if you want reach the global minima of the function, we need to go in the opposite direction of the steepest gradient and that is precisely what the gradient descent algorithm does.
The gradient descent algorithm will shift the parameters of the function iteratively based on the gradient of the function in each iteration eventually leading us to the global minima.

The gradient of a multivariate function such as the one in the example is given by the matrix
[[du/dx, du/dy],[dv/dx, dv/dy]]
The matrix is called as the Jacobian matrix and is of size n*m where n is the number of outputs and m is the size of the input.
So, now let us illustrate the function using a computation graph.

Let us first compute the derivative of 'u' with respect to each of the inputs. 
Let w = 2x
    h = 2y
	g = wh
Then u = sin(g). According to the chain rule,
du/dx = du/dg * dg/dw * dw/dx
Similarly,
du/dy = du/dg * dg/dh * dh/dy
Let z = w + h
v = cos(z)
dv/dx = dv/dz * dz/dw * dw/dx
dv/dy = dv/dz * dz/dh * dh/dy

The derivatives that are computed in each block in the graph is shown in a box with a red outline just next to the node. They are called the local derivatives. 
The local derivatives are multiplied as the computation flows through the graph from the input nodes x and y to the output nodes u and v.
Hence at the output node, one can get the derivative of the entire function using the chain rule.
You can also compute the derivative of the function by starting from the final node and working your way towards the input node. This is simply called the
backpropogation while the former way of computing the derivative is called the forward mode of automatic differentiation.
In the forward mode of the automatice differentiation, you start from the passing the first input variable (here 'x') and run it through the compuation
graph, computing the local derivative at each stage and multiplying it with the local derivative of the previous stage.
Finally, at the output nodes, you get the derivative of each of the outputs with respect to x.
Similary you repeat the procedure by passing y to get the partial derivative of the outputs with respect to y.
In the forward mode of automatic differentiation, for each dimension of the input, you calculate a row in the Jacobian matrix. Hence in
the forward mode of automatic differentiation you have to do m number of passes to compute the entire Jacobian matrix where m is the size of the input. 
In the case of  backpropogation, you run both the dimensions of the input and you calculate u and v at the end of the computation graph.
You then compute the derivatives starting from the output nodes and working your way to the input nodes.
Here, when you finally reach the input node, you calculate the derivative of a single output with repect to each of the input nodes. Hence in 
backpropogation, you have to do n number of backward passes to compute the Jacobian matrix.
Also, in the forward mode of automatic differentiation, the local derivative is always multiplied with the incoming derivative from the previous layers which is then propogated to the next layer. 
Likewise, in the case of backpropogation, the local derivative is multiplied with the incoming derivative coming from the higher layers before propogating it to the subsequent lower layers.
For example, let the partial derivative of u and v with respect to x be computed. 
In node w, A = dw/dx is computed. A is passed on to the next node g where A is multiplied with the local derivative dg/dw i.e B = dw/dx * dg.dw. B is then passed to the next node u where the local
derivative is du/dg. It is multipled with the incoming derivative B i.e C = du/dg * B = du/dg * dg/dw * dw/dx = du/dx
The partial derivative of u with respect to y is also computed similarly except for that in node g, dg/dh is computed as the incoming derivative for y comes from node h.
Well, of course one can argue why cannot compute the partial derivative of both the inputs in one pass in the forward mode of automatic differentiation or likewise
do the similar thing in the backward pass. The problem with that is you will have suffiencient memory to track the derivatives of each of the inputs. 
Unlike our example here, a network can have inputs consisting of thousands of dimensions.  

