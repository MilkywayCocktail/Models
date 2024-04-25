import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision.ops import generalized_box_iou_loss
from matplotlib.patches import Rectangle
import torch
from datetime import datetime
import os
import copy
from functools import wraps

#print(z * torch.transpose(z, -1, -2) - eye)
#print(torch.matmul(x, torch.transpose(x, -1, -2)))

a1 = np.array(['4th', '1st', '3rd', '2nd', '5th'])
samples = np.array([3,0,2,1,4])


class My:
    def __init__(self, x):
        self.x = x

    def wrapper(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            print(f"Starting {func.__name__}...")
            print("My x = ", self.x)
            ret = func(*args, **kwargs)
            print('Done')
            return ret

        return inner

    wrap = wrapper()

    @wrap
    def hello(self):
        print("Hello!")

        #self.hello = hello


my1 = My(55)
my1.hello()
my1.x = 99
my1.hello()
