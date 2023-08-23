import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision import datasets

y = [1,2,3,4,5,6]
y2 = [4,5,7,6,7,1]
l = list(range(len(y)))

keys = ['A', 'B', 'C']

dics = {key:[] for key in keys}
print(dics.items())

dic2 = {'A':2, 'B':3, 'C':4}
dics = {**dics, **dic2}

print(f"\r123445", end='')
print(f"\raabbc")

print(0.4 ** 10)