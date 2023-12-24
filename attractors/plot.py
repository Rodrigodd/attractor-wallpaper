"""
Plot the content of data.txt. data.txt should contains a '\n' separated list of
decimal numers.
"""

import matplotlib.pyplot as plt
import numpy as np

# Read data from file
with open('data.txt', 'r') as f:
    data = f.read().split('\n')
    data = [float(i) for i in data if i != '']
    
# Plot data
# plt.plot(data)

# Plot the data derivative, logaritmic scale
data = np.array(data)
data = np.abs(data[1:] - data[:-1])
plt.plot(np.log(data))

# Plot the exponential moving average of the data
exp = 0.03
def ema(array):
    ea = array[0]
    return np.array([
        ea := exp * x + (1 - exp) * ea
        for x in array
    ])

plt.plot(np.log(ema(data)))

plt.show()
