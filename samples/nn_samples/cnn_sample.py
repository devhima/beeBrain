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
sys.path.append('../../') #this to include the parent directory
import beeBrain.neuralNetworks as nn
import numpy as np

np.random.seed(1)
f = np.load('./datasets/cnn_dataset01.npz')
train_x, train_y = f['x_train'][:, :, :, None], f['y_train'][:, None]
test_x, test_y = f['x_test'][:2000][:, :, :, None], f['y_test'][:2000]

train_loader = nn.DataLoader(train_x, train_y, batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_layers = self.sequential(
            nn.layers.Conv2D(1, 6, (5, 5), (1, 1), "same", channels_last=True),  # => [n,28,28,6]
            nn.layers.MaxPool2D(2, 2),  # => [n, 14, 14, 6]
            nn.layers.Conv2D(6, 16, 5, 1, "same", channels_last=True),  # => [n,14,14,16]
            nn.layers.MaxPool2D(2, 2),  # => [n,7,7,16]
            nn.layers.Flatten(),  # => [n,7*7*16]
            nn.layers.Dense(7 * 7 * 16, 10, )
        )

    def forward(self, x):
        o = self.seq_layers.forward(x)
        return o


cnn = CNN()
opt = nn.optim.Adam(cnn.params, 0.001)
loss_fn = nn.losses.SparseSoftMaxCrossEntropyWithLogits()


for step in range(300):
    bx, by = train_loader.next_batch()
    by_ = cnn.forward(bx)
    loss = loss_fn(by_, by)
    cnn.backward(loss)
    opt.step()
    if step % 50 == 0:
        ty_ = cnn.forward(test_x)
        acc = nn.metrics.accuracy(np.argmax(ty_.data, axis=1), test_y)
        print("Step: %i | loss: %.3f | acc: %.2f" % (step, loss.data, acc))

