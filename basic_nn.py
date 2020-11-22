import numpy as np

def sigmoid(x):
    """
    Activation function
    f(x) = 1 / (1 e^(-x))
    """

    return 1 / (1 + np.exp(-x))

class Neuron:
    """
    A basic neuron, meant to be implemented in a basic Neural Network
    """
    
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        """
        Weight inputs, add bias and use the activation function (sigmoid)
        """

        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0,1])   # w1 = 0, w2 = 1
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])        # x1 = 2, x2 = 3
# print(n.feedforward(x))     # 0.9990889488055994 // Neuron is activated



class NeuralNetwork:
    """
    Basic Neural Network with :
    2 inputs
    2 - Hidden Layer (h1, h2)
    3 - Output Layer (o1)

    All neurons have the same weights and bias :
    - w = [0, 1] 
    - b = 0
    """
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        output_h1 = self.h1.feedforward(x)
        output_h2 = self.h2.feedforward(x)

        output_o1 = self.o1.feedforward(np.array([output_h1, output_h2]))

        return output_o1

# Testing if our NN works
network = NeuralNetwork()
inputs = np.array([2,3])
# print(network.feedforward(x))  = 0.7216325609518421

def mse_loss(y_true, _pred):
    """
    Mean Squared Loss function
    """
    return ((y_true - y_pred) ** 2).mean()

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred)) # 0.5
