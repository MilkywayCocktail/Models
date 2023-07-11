import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision import datasets

y = [1,2,3,4,5,6]
y2 = [4,5,7,6,7,1]
l = list(range(len(y)))

mnist = datasets.MNIST(root='../../dataset/MNIST/',
                       train=True,
                       download=True)

print(mnist[0])