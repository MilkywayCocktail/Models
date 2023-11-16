import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

#print(z * torch.transpose(z, -1, -2) - eye)
#print(torch.matmul(x, torch.transpose(x, -1, -2)))

a1 = np.array(['4th', '1st', '3rd', '2nd'])

inds = np.random.choice([0, 1, 2, 3], 4, replace=False)
inds = np.sort(inds)

samples = np.array([3,0,2,1])

print(inds)
print(np.argsort(samples))
print(inds[np.argsort(samples)])
print(a1[inds[np.argsort(samples)]])
print(a1[samples])
print(a1[np.argsort(samples)])