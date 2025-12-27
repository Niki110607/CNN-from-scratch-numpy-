# CNN-from-scratch-numpy-
A CNN deep learning implementation using only numpy

## Project Overview
I built this project to really get a understanding of the math that is done behind those deep learning algorithms before using libraries like pytorch or tensorflow.

## Core Architecture
* **Forward propagation:** Sliding window kernels for feature extraction, decreasing the array size using a Max Pooling layer and calculating the final output with Fully Connected layers.
* **Backpropagation:** Calculated gradients for 4D tensors manually.
* **Activations:** Implemented ReLU and Softmax functions.

## Optimization and Regularization
* **Adam Optimizer:** Faster convergence and a higher likelihood to find better local minima.
* **Dropout Neurons:** While training some neurons are getting deactivated to prevent overfitting.
* **Im2col Optimization:** Transforms a 4D tensor into 2D matrices to replace slow nested for loops with highly optimized numpy matrix multiplication.
  
## Performance
* **Dataset:** MNIST Digits
* **Accuracy:** 98.5%
* **Loss Function:** Cross-Entropy
