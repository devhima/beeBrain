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
[initializers.py]
- This file contains the implementation of initializing data(eg. synapses, weights, bias).
'''

import numpy as np

class BaseInitializer:
    def initialize(self, x):
        raise NotImplementedError


class RandomNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)


class RandomUniform(BaseInitializer):
    def __init__(self, low=0., high=1.):
        self._low = low
        self._high = high

    def initialize(self, x):
        x[:] = np.random.uniform(self._low, self._high, size=x.shape)


class Zeros(BaseInitializer):
    def initialize(self, x):
        x[:] = np.zeros_like(x)


class Ones(BaseInitializer):
    def initialize(self, x):
        x[:] = np.ones_like(x)


class TruncatedNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)
        truncated = 2*self._std + self._mean
        x[:] = np.clip(x, -truncated, truncated)


class Constant(BaseInitializer):
    def __init__(self, v):
        self._v = v

    def initialize(self, x):
        x[:] = np.full_like(x, self._v)


random_normal = RandomNormal()
random_uniform = RandomUniform()
zeros = Zeros()
ones = Ones()
truncated_normal = TruncatedNormal()
