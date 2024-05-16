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


def wrap_with_attributes(cls):
    def decorator(func):
        # Create a wrapper function that takes the class attributes as parameters
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use the attributes from the given class
            print(f"My x = {cls.x}:")

            # Call the original function
            return func(*args, **kwargs)

        return wrapper
    return decorator


class My:
    def __init__(self, x):
        self.x = x

        @wrap_with_attributes(self)
        def hello():
            print("Hello!")

        self.hello = hello

so = np.array([[0,1], [0,2], [1,2]])
tar = np.array([0, 2])
take, ind = tar[0], tar[-1]
_ind = np.where(so[:, 1] == ind)
print(so[_ind])
_take = np.where(so[_ind][:, 0] == take)
print(_take)
print(so[_ind][_take])
