#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Define bin edges every 10 units from 0 to 100
bins = np.arange(0, 110, 10)

# Create the histogram
plt.hist(student_grades, bins=bins, edgecolor='black')

# Label the axes
plt.xlabel('Grades')
plt.ylabel('Number of Students')

# Set the title
plt.title('Project A')

# Display the plot
plt.show()
