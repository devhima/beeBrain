'''
beeBrain - An Artificial Intelligence & Machine Learning library
by Dev. Ibrahim Said Elsharawy (www.devhima.tk)
'''

''''
MIT License

Copyright (c) 2018 Ibrahim Said Elsharawy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from numpy import exp, array, random, dot
from activationFunctions import ActivationFunctions

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        self.n_neurons = number_of_neurons
        self.n_inputs = number_of_inputs_per_neuron
        

class NeuralNetwork():
    def __init__(self, __layers, __activation_function):
        self.layers = __layers
        self.__activation_function = __activation_function

    # Activation function builder
    def build_activation_function(self, x, derivative):
        if self.__activation_function == ActivationFunctions.IDENTITY:
            # IDENTITY
            if derivative == True:
                return ActivationFunctions.identity_derivative(x)
            else:
                return ActivationFunctions.identity(x)
        elif self.__activation_function == ActivationFunctions.BINARY:
            # BINARY
            if derivative == True:
                return ActivationFunctions.binary_derivative(x)
            else:
                return ActivationFunctions.binary(x)
        elif self.__activation_function == ActivationFunctions.SIGMOID:
            # SIGMOID
            if derivative == True:
                return ActivationFunctions.sigmoid_derivative(x)
            else:
                return ActivationFunctions.sigmoid(x)
        elif self.__activation_function == ActivationFunctions.SIGNUM:
            # SIGNUM
            if derivative == True:
                return ActivationFunctions.signum_derivative(x)
            else:
                return ActivationFunctions.signum(x)
        elif self.__activation_function == ActivationFunctions.TANH:
            # TANH
            if derivative == True:
                return ActivationFunctions.tanh_derivative(x)
            else:
                return ActivationFunctions.tanh(x)
        elif self.__activation_function == ActivationFunctions.ARCTAN:
            # ARCTAN
            if derivative == True:
                return ActivationFunctions.arctan_derivative(x)
            else:
                return ActivationFunctions.arctan(x)
        elif self.__activation_function == ActivationFunctions.ISRU:
            # ISRU
            if derivative == True:
                return ActivationFunctions.isru_derivative(x)
            else:
                return ActivationFunctions.isru(x)
        elif self.__activation_function == ActivationFunctions.RELU:
            # RELU
            if derivative == True:
                return ActivationFunctions.relu_derivative(x)
            else:
                return ActivationFunctions.relu(x)
        elif self.__activation_function == ActivationFunctions.LRELU:
            # Leaky RELU
            if derivative == True:
                return ActivationFunctions.lrelu_derivative(x)
            else:
                return ActivationFunctions.lrelu(x)
        elif self.__activation_function == ActivationFunctions.PRELU:
            # PRELU
            if derivative == True:
                return ActivationFunctions.prelu_derivative(x)
            else:
                return ActivationFunctions.prelu(x)
        elif self.__activation_function == ActivationFunctions.RRELU:
            # RRELU
            if derivative == True:
                return ActivationFunctions.rrelu_derivative(x)
            else:
                return ActivationFunctions.rrelu(x)

    # Returns the result of activation function
    def activation_function(self, x):
        return self.build_activation_function(x, False)

    # Returns the result of the derivative of activation function
    # It indicates how confident we are about the existing weight.
    def activation_function_derivative(self, x):
        return self.build_activation_function(x, True)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            outputs = self.think(training_set_inputs)
            outputs_size = len(outputs)

            # Calculate the error for last layer (The difference between the desired output
            # and the predicted output).
            last_layer_error = training_set_outputs - outputs[outputs_size - 1]
            last_layer_delta = last_layer_error * self.activation_function_derivative(outputs[outputs_size - 1])
            layers_delta = []
            layers_delta.append(last_layer_delta)

            # Calculate the error for other layers (By looking at their weights,
            # we can determine by how much every layer contributed to the error in last layer).
            current_layer = outputs_size - 1
            for layer_out in reversed(outputs):
                if current_layer != outputs_size - 1:
                    last_layer_error = last_layer_delta.dot(self.layers[current_layer+1].synaptic_weights.T)
                    last_layer_delta = last_layer_error * self.activation_function_derivative(layer_out)
                    layers_delta.append(last_layer_delta)
                current_layer-=1

            # Calculate how much to adjust the weights by
            layers_adjustment = []
            layers_delta = reversed(layers_delta)
            last_training_set = training_set_inputs
            layer_num = 0
            for layer_delta in layers_delta:
                last_layer_adjustment = last_training_set.T.dot(layer_delta)
                last_layer_adjustment *= learning_rate
                last_training_set = outputs[layer_num]
                layer_num+=1
                layers_adjustment.append(last_layer_adjustment)

            # Adjust the weights.
            for i in xrange(0, len(self.layers)-1):
                self.layers[i].synaptic_weights += layers_adjustment[i]

    # The neural network thinks.
    def think(self, inputs):
        outputs = []
        last_inputs = inputs
        for layer in self.layers:
            last_inputs = self.activation_function(dot(last_inputs, layer.synaptic_weights))
            outputs.append(last_inputs)
        return outputs

    # The neural network prints its weights
    def print_weights(self):
        i = 1
        for layer in self.layers:
            print "Layer {} ({} neurons, with {} inputs)".format(i, layer.n_neurons, layer.n_inputs)
            print layer.synaptic_weights
            i+=1
