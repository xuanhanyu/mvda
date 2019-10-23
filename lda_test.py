import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
np.random.seed(seed=42)
# Create data
rectangles = np.array([[1, 1.5, 1.7, 1.45, 1.1, 1.6, 1.8], [1.8, 1.55, 1.45, 1.6, 1.65, 1.7, 1.75]])
triangles = np.array([[0.1, 0.5, 0.25, 0.4, 0.3, 0.6, 0.35, 0.15, 0.4, 0.5, 0.48],
                      [1.1, 1.5, 1.3, 1.2, 1.15, 1.0, 1.4, 1.2, 1.3, 1.5, 1.0]])
circles = np.array([[1.5, 1.55, 1.52, 1.4, 1.3, 1.6, 1.35, 1.45, 1.4, 1.5, 1.48, 1.51, 1.52, 1.49, 1.41, 1.39, 1.6,
                     1.35, 1.55, 1.47, 1.57, 1.48,
                     1.55, 1.555, 1.525, 1.45, 1.35, 1.65, 1.355, 1.455, 1.45, 1.55, 1.485, 1.515, 1.525, 1.495, 1.415,
                     1.395, 1.65, 1.355, 1.555, 1.475, 1.575, 1.485],
                    [1.3, 1.35, 1.33, 1.32, 1.315, 1.30, 1.34, 1.32, 1.33, 1.35, 1.30, 1.31, 1.35, 1.33, 1.32, 1.315,
                     1.38, 1.34, 1.28, 1.23, 1.25, 1.29,
                     1.35, 1.355, 1.335, 1.325, 1.3155, 1.305, 1.345, 1.325, 1.335, 1.355, 1.305, 1.315, 1.355, 1.335,
                     1.325, 1.3155, 1.385, 1.345, 1.285, 1.235, 1.255, 1.295]])
# Plot the data
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(111)
ax0.scatter(rectangles[0], rectangles[1], marker='s', c='grey', edgecolor='black')
ax0.scatter(triangles[0], triangles[1], marker='^', c='yellow', edgecolor='black')
ax0.scatter(circles[0], circles[1], marker='o', c='blue', edgecolor='black')

# Calculate the mean vectors per class
# Creates a 2x1 vector consisting of the means of the dimensions
mean_rectangles = np.mean(rectangles, axis=1).reshape(2, 1)
mean_triangles = np.mean(triangles, axis=1).reshape(2, 1)
mean_circles = np.mean(circles, axis=1).reshape(2, 1)
# Calculate the scatter matrices for the SW (Scatter within) and sum the elements up
scatter_rectangles = np.dot((rectangles - mean_rectangles), (rectangles - mean_rectangles).T)
scatter_triangles = np.dot((triangles - mean_triangles), (triangles - mean_triangles).T)
scatter_circles = np.dot((circles - mean_circles), (circles - mean_circles).T)
# Mind that we do not calculate the covariance matrix here because then we have to divide by n or n-1 as shown below
# print((1/7)*np.dot((rectangles-mean_rectangles),(rectangles-mean_rectangles).T))
# print(np.var(rectangles[0],ddof=0))
# Calculate the SW by adding the scatters within classes
SW = scatter_triangles + scatter_circles + scatter_rectangles
print(scatter_triangles.shape, scatter_circles.shape, scatter_rectangles.shape)
plt.show()
