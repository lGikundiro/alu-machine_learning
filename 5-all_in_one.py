#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Dataset 1: Cubic values
y0 = np.arange(0, 11) ** 3

# Dataset 2: Men's Height vs Weight
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

# Dataset 3: Exponential Decay of C-14
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

# Dataset 4: Exponential Decay of Radioactive Elements
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

# Dataset 5: Student Grades
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create a 3x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
fig.suptitle('All in One', fontsize=16)

# Plot 1: Cubic values
axs[0, 0].plot(np.arange(0, 11), y0, color='red')
axs[0, 0].set_title('Cubic Values')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')

# Plot 2: Men's Height vs Weight
axs[0, 1].scatter(x1, y1, color='magenta', s=10)
axs[0, 1].set_title("Men's Height vs Weight")
axs[0, 1].set_xlabel('Height (in)')
axs[0, 1].set_ylabel('Weight (lbs)')

# Plot 3: Exponential Decay of C-14
axs[1, 0].plot(x2, y2)
axs[1, 0].set_title('Exponential Decay of C-14')
axs[1, 0].set_xlabel('Time (years)')
axs[1, 0].set_ylabel('Fraction Remaining')
axs[1, 0].set_yscale('log')
axs[1, 0].set_xlim(0, 28650)

# Plot 4: Exponential Decay of Radioactive Elements
axs[1, 1].plot(x3, y31, 'r--', label='C-14')
axs[1, 1].plot(x3, y32, 'g-', label='Ra-226')
axs[1, 1].set_title('Exponential Decay of Radioactive Elements')
axs[1, 1].set_xlabel('Time (years)')
axs[1, 1].set_ylabel('Fraction Remaining')
axs[1, 1].set_xlim(0, 20000)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].legend(loc='upper right')

# Plot 5: Histogram of Student Grades
bins = np.arange(0, 110, 10)
axs[2, 0].hist(student_grades, bins=bins, edgecolor='black')
axs[2, 0].set_title('Project A')
axs[2, 0].set_xlabel('Grades')
axs[2, 0].set_ylabel('Number of Students')

# Hide the unused subplot (bottom right)
axs[2, 1].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
