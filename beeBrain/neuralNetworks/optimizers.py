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
[optimizers.py]
- This file contains the implementation of neural network training and optimizing process.
'''

import numpy as np

class Optimizer:
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self.vars = []
        self.grads = []
        for layer_p in self._params.values():
            for p_name in layer_p["vars"].keys():
                self.vars.append(layer_p["vars"][p_name])
                self.grads.append(layer_p["grads"][p_name])

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params=params, lr=lr)

    def step(self):
        for var, grad in zip(self.vars, self.grads):
            var -= self._lr * grad


class Momentum(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9):
        super().__init__(params=params, lr=lr)
        self._momentum = momentum
        self._mv = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, mv in zip(self.vars, self.grads, self._mv):
            dv = self._lr * grad
            mv[:] = self._momentum * mv + dv
            var -= mv


class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-06):
        super().__init__(params=params, lr=lr)
        self._eps = eps
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, v in zip(self.vars, self.grads, self._v):
            v += np.square(grad)
            var -= self._lr * grad / np.sqrt(v + self._eps)


class Adadelta(Optimizer):
    def __init__(self, params, lr=1., rho=0.9, eps=1e-06):
        super().__init__(params=params, lr=lr)
        self._rho = rho
        self._eps = eps
        self._m = [np.zeros_like(v) for v in self.vars]
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            v[:] = self._rho * v + (1. - self._rho) * np.square(grad)
            delta = np.sqrt(m + self._eps) / np.sqrt(v + self._eps) * grad
            var -= self._lr * delta
            m[:] = self._rho * m + (1. - self._rho) * np.square(delta)


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08):
        super().__init__(params=params, lr=lr)
        self._alpha = alpha
        self._eps = eps
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, v in zip(self.vars, self.grads, self._v):
            v[:] = self._alpha * v + (1. - self._alpha) * np.square(grad)
            var -= self._lr * grad / np.sqrt(v + self._eps)


class Adam(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params=params, lr=lr)
        self._betas = betas
        self._eps = eps
        self._m = [np.zeros_like(v) for v in self.vars]
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        b1, b2 = self._betas
        b1_crt, b2_crt = b1, b2
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            m[:] = b1 * m + (1. - b1) * grad
            v[:] = b2 * v + (1. - b2) * np.square(grad)
            b1_crt, b2_crt = b1_crt * b1, b2_crt * b2   # bias correction
            m_crt = m / (1. - b1_crt)
            v_crt = v / (1. - b2_crt)
            var -= self._lr * m_crt / np.sqrt(v_crt + self._eps)


class AdaMax(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params=params, lr=lr)
        self._betas = betas
        self._eps = eps
        self._m = [np.zeros_like(v) for v in self.vars]
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        b1, b2 = self._betas
        b1_crt = b1
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            m[:] = b1 * m + (1. - b1) * grad
            v[:] = np.maximum(b2 * v, np.abs(grad))
            b1_crt = b1_crt * b1  # bias correction
            m_crt = m / (1. - b1_crt)
            var -= self._lr * m_crt / (v + self._eps)