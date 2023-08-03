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


dic = {1: 1, 2: 2, 3: 3, 4: 4}
for va in dic.values():
    va = va + 2

print(dic.items())