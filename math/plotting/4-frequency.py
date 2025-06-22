#!user/bin/evn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68,15,50)

#plot the histogram
plt.hist(student_grades, bins=range(0,101,10), edgecolor="black")

#labeling the axix and adding the title
plt.xlabel('Grades')
plt.ylabel('Number of students')
plt.title('Project A')

#show the plot
plt.show()