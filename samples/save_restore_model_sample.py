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
x = np.linspace(-1, 1, 100)[:, None]       # [batch, 1]
y = x ** 2 + np.random.normal(0., 0.1, (100, 1))     # [batch, 1]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.layers.Dense(1, 5, nn.act.tanh)
        self.out = nn.layers.Dense(5, 1, nn.act.sigmoid)

    def forward(self, x):
        x = self.l1(x)
        o = self.out(x)
        return o


net1 = Net()
opt = nn.optim.Adam(net1.params, lr=0.1)
loss_fn = nn.losses.MSE()

for _ in range(1000):
    o = net1.forward(x)
    loss = loss_fn(o, y)
    net1.backward(loss)
    opt.step()
    print(loss)

# save net1 and restore to net2
net1.save("./mymodel.bbm")
net2 = Net()
net2.restore("./mymodel.bbm")
o2 = net2.forward(x)

plt.scatter(x, y, s=20)
plt.plot(x, o2.data, c="red", lw=3)
plt.show()

