import numpy as np
class ConvLayer:
    def __init__(self, in_channels, filter_amount, filter_size, learning_rate, adam):
        self.in_channels = in_channels
        self.filter_amount = filter_amount
        self.filter_size = filter_size
        self.out_channels = self.filter_amount
        self.padding = self.filter_size // 2
        self.learning_rate = learning_rate

        #initiating filters and biases
        self.filters = np.random.randn(self.out_channels, self.in_channels, self.filter_size, self.filter_size) / self.filter_size ** 2
        self.biases = np.zeros((1, self.out_channels, 1, 1))

        #if adam optimizer is used initiating adam
        self.adam = adam
        if self.adam:
            self._init_adam()



    def _init_adam(self):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.t = 0
        self.m_filters = np.zeros_like(self.filters)
        self.v_filters = np.zeros_like(self.filters)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def _adam(self):
        #adam momentum and variance changes the gradient values
        self.t += 1
        self.m_filters = self.beta_1 * self.m_filters + (1-self.beta_1) * self.filters_grad
        self.m_biases = self.beta_1 * self.m_biases + (1-self.beta_1) * self.biases_grad

        self.v_filters = self.beta_2 * self.v_filters + (1-self.beta_2) * self.filters_grad**2
        self.v_biases= self.beta_2 * self.v_biases + (1-self.beta_2) * self.biases_grad**2

        m_hat_filters = self.m_filters / (1 - self.beta_1 ** self.t)
        m_hat_biases = self.m_biases / (1 - self.beta_1 ** self.t)
        v_hat_filters = self.v_filters / (1 - self.beta_2 ** self.t)
        v_hat_biases = self.v_biases / (1 - self.beta_2 ** self.t)

        self.filters_grad = m_hat_filters / (np.sqrt(v_hat_filters) + 1e-8)
        self.biases_grad = m_hat_biases / (np.sqrt(v_hat_biases) + 1e-8)
        return self



    def _im2col(self, input):
        #the input array gets transformed into a column form with which the output can be calculated with a single dot product
        input_col = np.zeros((self.batch_size, self.in_channels, self.filter_size, self.filter_size, self.h, self.w))
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                input_col[:, :, i, j, :, :] = input[:, :, i:i+self.h, j:j+self.w]

        input_col = input_col.transpose(0, 4, 5, 1, 2, 3).reshape(self.batch_size * self.h * self.w, -1)
        return input_col

    def _flatten_filters(self, flip=False):
        #the filters are getting flattened to be used with the column arrays
        filters_new = self.filters.copy()
        if flip:
            filters_new = np.flip(self.filters, (2, 3))
            return filters_new.reshape(( -1))
        flattened_filters = filters_new.reshape((self.out_channels, -1))
        return flattened_filters

    def _col2im(self, output_col):
        #the calculated output gets reshaped into the image form
        output = output_col.reshape((self.out_channels, self.batch_size, self.h, self.w)).transpose(1, 0, 2, 3)
        return output



    def _relu(self, z):
        #relu outputs 0 if z < 0 or z if z > 0
        return np.maximum(0, z)

    def _relu_grad(self, z):
        #the relu gradient outputs 0 if z < 0 and 1 if z > 0
        return z > 0



    def forward(self, input, training):
        self.batch_size, self.in_channels, self.h, self.w = input.shape

        #padding of the input
        self.input_pad = np.pad(input,
                       ((0, 0),
                        (0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding)),
                        'constant',
                       constant_values=0)

        #here the convolution between the input and the filters is getting calculated
        self.input_col = self._im2col(self.input_pad)
        self.flattened_filters = self._flatten_filters()
        output_col = np.dot(self.flattened_filters, self.input_col.T)
        output = self._col2im(output_col)

        self.z = output + self.biases
        output = self._relu(self.z)
        return output



    def _col2im_back(self, input_grad_col):
        #the input gradient gets transformed back to the padded input shape
        input_grad_col_reshaped = input_grad_col.reshape(self.batch_size, self.h, self.w, self.in_channels, self.filter_size, self.filter_size)
        input_grad_col_reshaped = input_grad_col_reshaped.transpose(0, 3, 4, 5, 1, 2)

        padded_h = self.h + 2 * self.padding
        padded_w = self.w + 2 * self.padding
        input_grad = np.zeros((self.batch_size, self.in_channels, padded_h, padded_w))

        for i in range(self.filter_size):
            for j in range(self.filter_size):
                input_grad[:, :, i:i + self.h, j:j + self.w] += input_grad_col_reshaped[:, :, i, j, :, :]

        return input_grad

    def backward(self, output_grad, y):
        #gradient of the relu function
        z_grad = output_grad * self._relu_grad(self.z)
        #bias gradients are getting calculated and reshaped
        self.biases_grad = np.sum(z_grad, axis=(0, 2, 3))
        self.biases_grad = self.biases_grad.reshape((1, self.out_channels, 1, 1)) / self.batch_size

        #the gradient of the filters are evaluated by a convolution between the input and the pre-relu gradients
        #the method used for the convolution is pretty much the same as the forward propagation
        z_grad_reshaped = z_grad.transpose(1, 0, 2, 3).reshape((self.out_channels, -1))
        filters_grad_flattened = np.dot(z_grad_reshaped, self.input_col)
        self.filters_grad = filters_grad_flattened.reshape((self.filters.shape)) / self.batch_size

        if self.adam:
            self._adam()

        #gradient descend
        self.filters -= self.learning_rate * self.filters_grad
        self.biases -=  self.learning_rate * self.biases_grad

        #computation of the gradient with respect to the input
        input_grad_col = np.dot(z_grad_reshaped.T, self.flattened_filters)
        input_grad_pad = self._col2im_back(input_grad_col)

        #remove padding
        input_grad = input_grad_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return input_grad