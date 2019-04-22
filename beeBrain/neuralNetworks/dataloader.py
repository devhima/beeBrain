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
[dataloader.py]
- This file contains implementation of loading & preparing training data from given dataset.
'''

import numpy as np

class DataLoader:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.bs = batch_size
        self.p = 0
        self.bg = self.batch_generator()

    def batch_generator(self):
        while True:
            p_ = self.p + self.bs
            if p_ > len(self.x):
                self.p = 0
                continue
            if self.p == 0:
                indices = np.random.permutation(len(self.x))
                self.x[:] = self.x[indices]
                self.y[:] = self.y[indices]
            bx = self.x[self.p:p_]
            by = self.y[self.p:p_]
            self.p = p_
            yield bx, by

    def next_batch(self):
        return next(self.bg)
