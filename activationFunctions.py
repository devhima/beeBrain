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

from numpy import exp
import numpy
import enum

class ActivationFunctions(enum.Enum):
    
    # >>>> Identity function <<<<
    # The Identity function, is a function that always returns
    # the same value that was used as its argument.
    # In equations, the function is given by f(x) = x.
    @staticmethod
    def identity(x):
        return x
    
    # The derivative of the Identity function, by calculating the first derivative of x.
    # It always returns 1
    @staticmethod
    def identity_derivative(x):
        return 1
    
    #Enum name&value
    IDENTITY = 0

    #____________________________________________________________
    
    # >>>> Binary step (Heaviside step) function <<<<
    # The Binary function, is a function that returns
    # value is 0 for negative argument and 1 for positive argument.
    @staticmethod
    def binary(x):
        return numpy.where(x>=0, 1, 0)
    
    # The derivative of the Binary step (Heaviside step) function, you can get it
    # by calculating the derivative of dH/dx that equals the dirac delta of x.
    # It returns 0 if x>0 or x<0, and it returns ? if x = 0.
    @staticmethod
    def binary_derivative(x):
        return 0
    
    #Enum name&value
    BINARY = 1

    #____________________________________________________________

    # >>>> Sigmoid function <<<<
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    #Enum name&value
    SIGMOID = 2

    #____________________________________________________________

    # >>>> Signum (Softsign) function <<<<
    # The signum function, which extracts the sign of a real number x
    # We pass the weighted sum of the inputs through this function to
    # normalise them between -1 and +1.
    @staticmethod
    def signum(x):
        y = x / (1 + abs(x))
        return y / abs(y)

    # The derivative of the signum function, by calculating abs(signum(x))
    # It returns 1 or 0
    @staticmethod
    def signum_derivative(x):
        y = 1 / ((1 + abs(x)) ** 2)
        return y / abs(y)
        
    #Enum name&value
    SIGNUM = 3

    #____________________________________________________________
    
    # >>>> TanH (Hyperbolic tangent) function <<<<
    # The hyperbolic tangent is the solution to the differential equation 
    # f'=1-f^2 with f(0)=0 and the nonlinear boundary value problem
    @staticmethod
    def tanh(x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    # The derivative of the TanH function, by calculating 1-(f(x))^2
    @staticmethod
    def tanh_derivative(x):
        return 1 - (x ** 2)
        
    #Enum name&value
    TANH = 4

    #____________________________________________________________

