import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision.ops import generalized_box_iou_loss
from matplotlib.patches import Rectangle
import torch
from datetime import datetime

#print(z * torch.transpose(z, -1, -2) - eye)
#print(torch.matmul(x, torch.transpose(x, -1, -2)))

a1 = np.array(['4th', '1st', '3rd', '2nd', '5th'])
samples = np.array([3,0,2,1,4])

inds = np.random.choice([0, 1, 2, 3, 4], 4, replace=False)
inds = np.sort(inds)

s = samples[inds]
#print(s)
#print(a1[inds[np.argsort(s)]])

arrr = np.array([ 129, 2223, 3198, 3975, 5332, 5432, 6340, 6361])

*first, last = arrr

print(first)
print(last)

params = {}

if params:
    print("Aru")
else:
    print("Nai")

params['weight'] = 0

if params:
    print("Aru")
else:
    print("Nai")
