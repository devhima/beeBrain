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
[saver.py]
- This file contains the implementation of saving or restoring neural network models using pickle library.
'''

import pickle
from beeBrain import neuralNetworks as nn

class Saver:
    @staticmethod
    def save(model, path):
        assert isinstance(model, nn.Module)
        vars = {name: p["vars"] for name, p in model.params.items()}
        with open(path, "wb") as f:
            pickle.dump(vars, f)

    @staticmethod
    def restore(model, path):
        assert isinstance(model, nn.Module)
        with open(path, "rb") as f:
            params = pickle.load(f)
        for name, param in params.items():
            for p_name in model.params[name]["vars"].keys():
                model.params[name]["vars"][p_name][:] = param[p_name]
                model.params[name]["vars"][p_name][:] = param[p_name]
