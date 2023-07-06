import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

y = [1,2,3,4,5,6]
y2 = [4,5,7,6,7,1]
l = list(range(len(y)))
x = np.array([0.1,0.01, 0.001])
x = -np.log(x)
norm = plt.Normalize(x.min(), x.max())
map_vir = cm.get_cmap(name='viridis')
c = map_vir(norm(x))
print(c)

plt.plot(l, y)
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(l, y2)
plt.axvline(3, linestyle='--', color=c[0], label=f'lr={x[0]}')
plt.axvline(5, linestyle='--', color=c[1], label=f'lr={x[1]}')
plt.axvline(6, linestyle='--', color=c[2], label=f'lr={x[1]}')
plt.legend()
plt.show()
