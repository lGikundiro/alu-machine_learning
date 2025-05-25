#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Define labels and colors
people = ['Farrah', 'Fred', 'Felicia']
fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Set up the x-axis positions
x = np.arange(len(people))
width = 0.5

# Initialize the bottom positions for stacking
bottom = np.zeros(len(people))

# Create the stacked bar chart
for i in range(len(fruit)):
    plt.bar(x, fruit[i], width, bottom=bottom, color=colors[i], label=fruit_labels[i])
    bottom += fruit[i]

# Customize the plot
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(x, people)
plt.yticks(np.arange(0, 81, 10))
plt.legend()

# Display the plot
plt.show()
