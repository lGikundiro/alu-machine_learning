#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# Create the plot
plt.plot(x, y)

# Set the y-axis to logarithmic scale
plt.yscale('log')

# Label the axes
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')

# Set the title
plt.title('Exponential Decay of C-14')

# Set the x-axis range
plt.xlim(0, 28650)

# Display the plot
plt.show()
