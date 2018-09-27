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

from numpy import array, random
from beeNeuralNetwork import *
from activationFunctions import ActivationFunctions

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (3 neurons, each with 2 inputs)
    layer1 = NeuronLayer(3, 2)

    # Create layer 2 (4 neuron with 3 inputs)
    layer2 = NeuronLayer(4, 3)
    
    # Create layer 3 (2 neuron with 4 inputs)
    layer3 = NeuronLayer(2, 4)
    
    # Create layer 3 (1 neuron with 2 inputs)
    layer4 = NeuronLayer(1, 2)
    
    netLayers = [layer1, layer2, layer3, layer4]

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(netLayers, ActivationFunctions.SIGMOID)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    # The training set. We have 3 examples, each consisting of 2 input values
    # and 1 output value.
    training_set_inputs = array([[30, 50], [40, 18], [8, 10]])
    training_set_outputs = array([[1, -1 , 1]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print "Stage 3) Considering a new situation [80, 50] -> ?: "
    output = neural_network.think(array([80, 50]))
    print output[len(output)-1]
