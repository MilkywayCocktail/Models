import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision.ops import generalized_box_iou_loss
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

sdf = ('a', 'b')

a, b = sdf
print(a, b)


def bbx_loss(bbx1, bbx2):
    # x, y, w, h to x1, y1, x2, y2
    bbx1[-1] = bbx1[-1] + bbx1[-3]
    bbx1[-2] = bbx1[-2] + bbx1[-4]
    bbx2[-1] = bbx2[-1] + bbx2[-3]
    bbx2[-2] = bbx2[-2] + bbx2[-4]
    return generalized_box_iou_loss(bbx1, bbx2)

print(bbx_loss(torch.tensor([1,1,1,1]), torch.tensor([1,1,2,1])))


