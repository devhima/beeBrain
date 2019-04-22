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

import sys
sys.path.append('../') #this to include the parent directory
import numpy as np
import beeBrain.neuralNetworks as nn
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.linspace(-1, 1, 200)[:, None]       # [batch, 1]
y = x ** 2 + np.random.normal(0., 0.1, (200, 1))     # [batch, 1]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        w_init = nn.init.RandomUniform()
        b_init = nn.init.Constant(0.1)

        self.l1 = nn.layers.Dense(1, 10, nn.act.tanh, w_init, b_init)
        self.l2 = nn.layers.Dense(10, 10, nn.act.tanh, w_init, b_init)
        self.out = nn.layers.Dense(10, 1, w_initializer=w_init, b_initializer=b_init)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o


net = Net()
opt = nn.optim.Adam(net.params, lr=0.1)
loss_fn = nn.losses.MSE()

for step in range(100):
    o = net.forward(x)
    loss = loss_fn(o, y)
    net.backward(loss)
    opt.step()
    print("Step: %i | loss: %.5f" % (step, loss.data))

plt.scatter(x, y, s=20)
plt.plot(x, o.data, c="red", lw=3)
plt.show()

