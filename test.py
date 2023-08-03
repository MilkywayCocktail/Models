import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision import datasets

y = [1,2,3,4,5,6]
y2 = [4,5,7,6,7,1]
l = list(range(len(y)))

st = 7.89
s = np.ceil(st).astype(int)
print(s)


my = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
print(np.array(my).reshape((-1, 2, 2)))

inds = np.random.choice(list(range(5)), 3)
print(inds)

li = np.array([1,2,3,4,5,6,7,8,9])
print(li[inds])