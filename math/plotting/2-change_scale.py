#!/user/bin/evn
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,21000,1000)
r = np.log(0.5)
t = 5730
y = np.exp((r/t) * x)

# Plot x ↦ y as a line graph
plt.plot(x,y)

# set x-axis range
plt.xlim(0,28650)

# set y-axis to logarithmic scale
plt.yscale('log')

#add the labels and title 
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')

#show the plot 
plt.show()