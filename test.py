import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

x = torch.rand(1, 9)
print(x)
eye = torch.eye(3)

z = x.view(1, 3, 3)
print(z)

#print(z * torch.transpose(z, -1, -2) - eye)
print(torch.matmul(x, torch.transpose(x, -1, -2)))