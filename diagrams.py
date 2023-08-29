import numpy as np
import matplotlib.pyplot as plt
prop = np.load('../results/0829prop.npy')
ae = np.load('../results/0829ae.npy')
ts = np.load('../results/0829ts.npy')
wi2vi = np.load('../results/0829wi2vi.npy')

sorted_prop = np.sort(prop)
yvals_prop = np.arange(len(sorted_prop))/float(len(sorted_prop))
sorted_ae = np.sort(ae)
yvals_ae = np.arange(len(sorted_ae))/float(len(sorted_ae))
sorted_ts = np.sort(ts)
yvals_ts = np.arange(len(sorted_ts))/float(len(sorted_ts))
sorted_wi2vi = np.sort(wi2vi) / 4.6875
yvals_wi2vi = np.arange(len(sorted_wi2vi))/float(len(sorted_wi2vi))

plt.plot(sorted_prop, yvals_prop, label='Proposed')
plt.plot(sorted_ae, yvals_ae, label='Encoder-Decoder')
plt.plot(sorted_ts, yvals_ts, label='Teacher-Student')
plt.plot(sorted_wi2vi, yvals_wi2vi, label='Wi2Vi')

# plt.xscale("log", base=2)
plt.legend()
plt.title('CDF of image loss (MSE)')
plt.xlabel('MSE per image')
plt.ylabel('Cumulative Freq')
plt.show()
