'''
beeBrain - An Artificial Intelligence & Machine Learning library
by Dev. Ibrahim Said Elsharawy (www.devhima.tk)
'''

''''
MIT License

Copyright (c) 2019 Ibrahim Said Elsharawy

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

'''
[activations.py]
- This file contains the implementation of the activation functions.
'''

import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Activation):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), np.full_like(x, self.alpha))


class ELU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(x, self.alpha*(np.exp(x)-1))

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), self.forward(x) + self.alpha)


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(np.tanh(x))


class Sigmoid(Activation):
    def forward(self, x):
        return 1./(1.+np.exp(-x))

    def derivative(self, x):
        f = self.forward(x)
        return f*(1.-f)


class SoftPlus(Activation):
    def forward(self, x):
        return np.log(1. + np.exp(x))

    def derivative(self, x):
        return 1. / (1. + np.exp(-x))


class SoftMax(Activation):
    def forward(self, x, axis=-1):
        shift_x = x - np.max(x, axis=axis, keepdims=True)   # stable softmax
        exp = np.exp(shift_x + 1e-6)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def derivative(self, x):
        return np.ones_like(x)


relu = ReLU()
leakyrelu = LeakyReLU()
elu = ELU()
tanh = Tanh()
sigmoid = Sigmoid()
softplus = SoftPlus()
softmax = SoftMax()

