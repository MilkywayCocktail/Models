import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

#print(z * torch.transpose(z, -1, -2) - eye)
#print(torch.matmul(x, torch.transpose(x, -1, -2)))

a1 = np.array(['4th', '1st', '3rd', '2nd', '5th'])
samples = np.array([3,0,2,1,4])

inds = np.random.choice([0, 1, 2, 3, 4], 4, replace=False)
inds = np.sort(inds)

s = samples[inds]
#print(s)
#print(a1[inds[np.argsort(s)]])

ones = np.ones((3, 3))

twos = np.ones((3, 3)) * 2

m = np.concatenate((ones[np.newaxis, ...], twos[np.newaxis, ...]), axis=0)
print(m.shape)
n = np.concatenate((m[0], m[1]), axis=-1)
print(n.shape)