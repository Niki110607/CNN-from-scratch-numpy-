import numpy as np

from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from FullyConnectLayer import FullyConnectLayer
class CNN:
    def __init__(self, structure, epochs, batch_size, learning_rate, colour_channels, picture_h, picture_w, adam=False, dropout=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.y_size = structure[-1][2]
        self.colour_channels = colour_channels
        self.picture_h = picture_h
        self.picture_w = picture_w

        #interpreting structure and creating the layers
        self.layers = []
        for layer in structure:
            if layer[0] == "conv":
                self.layers.append(ConvLayer(layer[1], layer[2], layer[3], learning_rate, adam))

            elif layer[0] == "pool":
                self.layers.append(PoolLayer(layer[1]))

            elif layer[0] == "FC":
                self.layers.append(FullyConnectLayer(layer[1], layer[2], learning_rate, layer[3], adam, dropout))



    def _create_mini_batches(self, X, y):
        #shuffling the data and creating mini batches for training
        rows = X.shape[0]
        shuffle = np.random.permutation(rows)
        mini_batches = []
        for i in range(0, rows, self.batch_size):
            shuffle_batch_indices = shuffle[i:i + self.batch_size]
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            mini_batches.append((X_batch, y_batch))
        return mini_batches

    def _vectorize(self, y):
        #transforming the y values from a number that states the outcome to a vector of zeros with a the value one on the yth place
        y_vector = np.zeros((len(y), self.y_size))
        y_vector[np.arange(len(y)), y] = 1
        return y_vector

    def _calc_loss(self, y_hat, y):
        #calculates cross entropy loss for classification
        log_likelihood = -np.log(y_hat[range(y.shape[0]), np.argmax(y, axis=1)] + 1e-8)
        return np.mean(log_likelihood)

    def fit(self, X, y):
        #reshaping X and y into the preferred forms
        X = X.reshape(X.shape[0], self.colour_channels, self.picture_h, self.picture_w)
        y = self._vectorize(y)
        for epoch in range(self.epochs):
            #create mini batches
            mini_batches = self._create_mini_batches(X, y)
            Loss_mini_batch = []
            for mini_batch in mini_batches:
                X_input, y_input = mini_batch

                #forward propagation
                for layer in self.layers[0::]:
                    X_input = layer.forward(X_input, training=True)
                y_hat = X_input

                #storing the loss
                Loss_mini_batch.append(self._calc_loss(y_hat, y_input))

                #backward propagation
                output_grad = 0
                for layer in reversed(self.layers[0::]):
                    output_grad = layer.backward(output_grad, y_input)
            print(epoch, np.mean(np.array(Loss_mini_batch)))
        return self



    def predict(self, X):
        #reshaping X
        X = X.reshape(X.shape[0], self.colour_channels, self.picture_h, self.picture_w)
        #calculating the output
        for layer in self.layers:
            X = layer.forward(X, training=False)
        #returning the value with the highest probability
        return np.argmax(X, axis=1)