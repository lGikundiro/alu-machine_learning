#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Data
np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))  # 4 rows (fruit types) x 3 columns (people)

# Bar positions and width
people = ['Farrah', 'Fred', 'Felicia']
bar_width = 0.5
x = np.arange(len(people))

# Plot each fruit type stacked on top of the previous
plt.bar(x, fruit[0], color='red', width=bar_width, label='Apples')
plt.bar(x, fruit[1], bottom=fruit[0], color='yellow', width=bar_width, label='Bananas')
plt.bar(x, fruit[2], bottom=fruit[0] + fruit[1], color='#ff8000', width=bar_width, label='Oranges')
plt.bar(x, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2], color='#ffe5b4', width=bar_width, label='Peaches')

# Add labels, legend, and title
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 81, 10))  # Set y-axis ticks from 0 to 80 with step 10
plt.title('Number of Fruit per Person')
plt.xticks(x, people)  # Set x-axis ticks and labels
plt.legend()  # Add legend

# Display the plot
plt.show()
