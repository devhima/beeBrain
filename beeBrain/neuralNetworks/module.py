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
[module.py]
- This file contains the implementation of the module (propagation & backpropagation) neural network module.
'''

import numpy as np
from beeBrain import neuralNetworks as nn

class Module(object):
    def __init__(self):
        self._ordered_layers = []
        self.params = {}

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss):
        assert isinstance(loss, nn.losses.Loss)
        # find net order
        layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, nn.layers.BaseLayer):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        last_layer = self._ordered_layers[-1]
        last_layer.data_vars["out"].set_error(loss.delta)
        for layer in self._ordered_layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, nn.layers.ParamLayer):
                for k in layer.param_vars.keys():
                    self.params[layer.name]["grads"][k][:] = grads[k]

    def save(self, path):
        saver = nn.Saver()
        saver.save(self, path)

    def restore(self, path):
        saver = nn.Saver()
        saver.restore(self, path)

    def sequential(self, *layers):
        assert isinstance(layers, (list, tuple))
        for i, l in enumerate(layers):
            self.__setattr__("layer_%i" % i, l)
        return SeqLayers(layers)

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, nn.layers.ParamLayer):
            layer = value
            self.params[key] = {
                "vars": layer.param_vars,
                "grads": {k: np.empty_like(layer.param_vars[k]) for k in layer.param_vars.keys()}
            }
        object.__setattr__(self, key, value)


class SeqLayers:
    def __init__(self, layers):
        assert isinstance(layers, (list, tuple))
        for l in layers:
            assert isinstance(l, nn.layers.BaseLayer)
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)