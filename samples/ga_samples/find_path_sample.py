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
from beeBrain.geneticAlgorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np

N_MOVES = 150
DNA_SIZE = N_MOVES*2         # 40 x moves, 40 y moves
DIRECTION_BOUND = [0, 1]
CROSS_RATE = 0.8
MUTATE_RATE = 0.0001
POP_SIZE = 100
N_GENERATIONS = 100
GOAL_POINT = [10, 5]
START_POINT = [0, 5]
OBSTACLE_LINE = np.array([[5, 2], [5, 8]])


class myGA(GeneticAlgorithm):
    
    def DNA2product(self, DNA, n_moves, start_point):
        pop = (DNA - 0.5) / 2
        pop[:, 0], pop[:, n_moves] = start_point[0], start_point[1]
        lines_x = np.cumsum(pop[:, :n_moves], axis=1)
        lines_y = np.cumsum(pop[:, n_moves:], axis=1)
        return lines_x, lines_y

    def get_fitness(self, lines_x, lines_y, goal_point, obstacle_line):
        dist2goal = np.sqrt((goal_point[0] - lines_x[:, -1]) ** 2 + (goal_point[1] - lines_y[:, -1]) ** 2)
        fitness = np.power(1 / (dist2goal + 1), 2)
        points = (lines_x > obstacle_line[0, 0] - 0.5) & (lines_x < obstacle_line[1, 0] + 0.5)
        y_values = np.where(points, lines_y, np.zeros_like(lines_y) - 100)
        bad_lines = ((y_values > obstacle_line[0, 1]) & (y_values < obstacle_line[1, 1])).max(axis=1)
        fitness[bad_lines] = 1e-6
        return fitness


class Line(object):
    def __init__(self, n_moves, goal_point, start_point, obstacle_line):
        self.n_moves = n_moves
        self.goal_point = goal_point
        self.start_point = start_point
        self.obstacle_line = obstacle_line

        plt.ion()

    def plotting(self, lines_x, lines_y):
        plt.cla()
        plt.scatter(*self.goal_point, s=200, c='r')
        plt.scatter(*self.start_point, s=100, c='b')
        plt.plot(self.obstacle_line[:, 0], self.obstacle_line[:, 1], lw=3, c='k')
        plt.plot(lines_x.T, lines_y.T, c='k')
        plt.xlim((-5, 15))
        plt.ylim((-5, 15))
        plt.pause(0.01)


ga = myGA(DNA_size=DNA_SIZE, DNA_bound=DIRECTION_BOUND,
        cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

env = Line(N_MOVES, GOAL_POINT, START_POINT, OBSTACLE_LINE)

for generation in range(N_GENERATIONS):
    lx, ly = ga.DNA2product(ga.pop, N_MOVES, START_POINT)
    fitness = ga.get_fitness(lx, ly, GOAL_POINT, OBSTACLE_LINE)
    ga.evolve(fitness)
    print('Gen:', generation, '| best fit:', fitness.max())
    env.plotting(lx, ly)

plt.ioff()
plt.show()


