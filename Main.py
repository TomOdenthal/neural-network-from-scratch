import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, nx, ny):
        n_layerneurons = 4
        self.input = np.zeros((1, nx))
        self.w1 = np.random.rand(nx, n_layerneurons)
        self.z1 = np.zeros((1, n_layerneurons))
        self.a1 = np.zeros((1, n_layerneurons))
        self.w2 = np.random.rand(n_layerneurons, 1)
        self.z2 = np.zeros((1, ny))
        self.output = np.zeros((1, ny))

    def feedforward(self, x):
        self.input = x
        self.z1 = np.dot(self.input, self.w1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.output = sigmoid(self.z2)
        return self.output

    def backprop(self, y):

        # calculate needed weight changes by derivative of the loss function
        dw2 = np.dot(self.a1.T, (2 * (y - self.output) * sigmoid_derivative(self.z2)))
        dw1 = np.dot(self.input.T, (np.dot(2 * (y - self.output) * sigmoid_derivative(self.z2), self.w2.T) * sigmoid_derivative(self.z1)))

        # update the weights
        self.w1 += dw1
        self.w2 += dw2


# Data for [ (a AND b) XOR c ]
x_train = np.array([[1, 1, 1],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 1]])
y_train = np.array([[0],
                    [0],
                    [1],
                    [1]])

x_test = np.array([[0, 0, 1]])
y_test = np.array([[1]])

# init Neural Network
myNN = NeuralNetwork(3, 1)

# training
for i in range(0, x_train.shape[0]):
    print('Iteration ' + str(i+1) + ' of ' + str(x_train.shape[0]))
    xt = x_train[i, :].reshape(1, x_train[i, :].size)
    yt = y_train[i, 0].reshape(1, y_train[i, :].size)
    y_out = myNN.feedforward(xt)
    loss = (y_out - yt)**2
    print('Output: ' + str(float(y_out[0, 0])) + ' True: ' + str(float(yt[0, 0])))
    print('Loss: ' + str(float(loss)))
    myNN.backprop(yt)
print('')

# testing
print('Testing:')
y_out = myNN.feedforward(x_test)
loss = (y_out - y_test)**2
print('Output: ' + str(float(y_out[0, 0])) + ' True: ' + str(float(y_test[0, 0])))
print('Loss: ' + str(float(loss)))

