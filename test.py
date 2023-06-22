import matplotlib.pyplot as plt

y = [1,2,3,4,5,6]
x = 6
z = 7

plt.plot(y)
plt.axvline(3, color='r', linestyle='--', label=f'lr={x}')
plt.axvline(5, color='r', linestyle='--', label=f'lr={z}')
plt.legend()
plt.show()
