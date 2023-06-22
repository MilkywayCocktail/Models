import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

y = [1,2,3,4,5,6]
x = np.array([6,19, 10])
norm = plt.Normalize(x.min(), x.max())
map_vir = cm.get_cmap(name='viridis')
c = map_vir(norm(x))
print(c)

plt.plot(y)
plt.axvline(3, linestyle='--', color=c[0], label=f'lr={x[0]}')
plt.axvline(5, linestyle='--', color=c[1], label=f'lr={x[1]}')
plt.axvline(6, linestyle='--', color=c[2], label=f'lr={x[1]}')
plt.legend()
plt.show()
