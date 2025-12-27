import numpy as np
#Max pooling layer takes the maximum pixel value from a square with length self.size and minimizes this square to one value
class PoolLayer:
    def __init__(self, size=2):
        self.size = size

    def forward(self, input, training):
        self.input = input
        batch_size, in_channels, h, w = input.shape
        h_new = h//self.size
        w_new = w//self.size

        #the input gets reshaped into a format, which is similar to the one used in the convolution layer, that increases the computational speed because it does not require nested for loops
        input_reshaped = input.reshape((batch_size, in_channels, h_new, self.size, w_new, self.size))
        input_transposed = input_reshaped.transpose(0, 1, 2, 4, 3, 5).copy()
        output = np.max(input_transposed, axis=(4, 5))

        self.input_transformed = input_transposed
        self.output = output
        return output

    def backward(self, output_grad, y):
        #first the gradient of the previous layer gets multiplied with a mask so only the values which actually affected the loss function (the max values) are getting gradients
        mask = (self.input_transformed == self.output[:, :, :, :, np.newaxis, np.newaxis])
        output_grad_expanded = output_grad[:, :, :, :, np.newaxis, np.newaxis]

        input_grad_transposed = mask * output_grad_expanded
        #the gradient array is getting transformed back to match the shape of the input
        input_grad_reshaped = input_grad_transposed.transpose(0, 1, 2, 4, 3, 5)
        input_grad = input_grad_reshaped.reshape((self.input.shape))
        return input_grad