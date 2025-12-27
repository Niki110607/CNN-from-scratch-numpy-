import numpy as np
#Fully connected Layer like the ones used in an ANN
class FullyConnectLayer:
    def __init__(self, in_size, out_size, learning_rate, softmax=False, adam=False, dropout=0):
        self.in_size = in_size
        self.out_size = out_size
        self.learning_rate = learning_rate
        self.softmax = softmax
        self.adam = adam
        self.dropout = dropout

        #initiating weights and biases
        self.weights = np.random.randn(self.in_size, self.out_size) * np.sqrt(2/self.in_size)
        self.biases = np.zeros(self.out_size)

        #if adam is used initiating adam
        if self.adam:
            self._init_adam()

    def _init_adam(self):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.t = 0
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def _adam(self):
        #adam momentum and variance changes the gradient values
        self.t += 1
        self.m_weights = self.beta_1 * self.m_weights + (1-self.beta_1) * self.weights_grad
        self.m_biases = self.beta_1 * self.m_biases + (1-self.beta_1) * self.biases_grad

        self.v_weights = self.beta_2 * self.v_weights + (1-self.beta_2) * self.weights_grad**2
        self.v_biases= self.beta_2 * self.v_biases + (1-self.beta_2) * self.biases_grad**2

        m_hat_weights = self.m_weights / (1 - self.beta_1 ** self.t)
        m_hat_biases = self.m_biases / (1 - self.beta_1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta_2 ** self.t)
        v_hat_biases = self.v_biases / (1 - self.beta_2 ** self.t)

        self.weights_grad = m_hat_weights / (np.sqrt(v_hat_weights) + 1e-8)
        self.biases_grad = m_hat_biases / (np.sqrt(v_hat_biases) + 1e-8)
        return self



    def _create_dropout(self):
        #dropout deactivates some neurons while training to prevent overfitting
        scale = 1 / (1 - self.dropout)
        dropout_mask = (np.random.random(self.output.shape) > self.dropout) * scale
        return dropout_mask



    def _relu(self, z):
        #relu outputs 0 if z < 0 or z if z > 0
        return np.maximum(0, z)

    def _relu_grad(self, z):
        #the relu gradient outputs 0 if z < 0 and 1 if z > 0
        return z > 0

    def _softmax(self, z):
        #softmax is used to transform a vector of float numbers to according probabilities which add up to 1
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)



    def forward(self, input, training):
        self.original_shape = input.shape
        #the input gets flattened
        self.input = input.reshape(input.shape[0], -1)
        #the weights and biases get applied
        self.z = np.dot(self.input, self.weights) + self.biases

        #if this is not the last layer thus does not use softmax the function applies the relu function and in case of training creates the neuron dropout
        if not self.softmax:
            self.output = self._relu(self.z)
            if training:
                self.mask = self._create_dropout()
                self.output *= self.mask
        #if this is the last layer it uses the softmax function
        else:
            self.output = self._softmax(self.z)
        return self.output

    def backward(self, output_grad, y):
        #the gradient of the cross-entropy loss function with respect of the pre-softmax output is just the difference between the output and the true y value
        if self.softmax:
            grad_z = self.output - y
        #this is the gradient of the relu function and the inversion of the dropout
        else:
            grad_z = output_grad * self._relu_grad(self.z)
            grad_z *= self.mask

        #the weights and biases gradients get calculated
        self.weights_grad = np.dot(self.input.T, grad_z) / self.original_shape[0]
        self.biases_grad = np.sum(grad_z, axis=0) / self.original_shape[0]

        #the gradient of the loss function with respect to the input gets handed to the back propagation of the previous layer
        input_grad = np.dot(grad_z, self.weights.T)
        input_grad = input_grad.reshape(self.original_shape)

        if self.adam:
            self._adam()

        #gradient descend
        self.weights -= self.learning_rate * self.weights_grad
        self.biases -= self.learning_rate * self.biases_grad
        return input_grad