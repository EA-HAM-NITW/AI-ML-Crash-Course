# Introduction to NumPy, Pandas, and Matplotlib

## 1. NumPy (Numerical Python)

NumPy is the fundamental package for scientific computing in Python. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures efficiently.

### Key Features:

- **N-dimensional Array**: Provides a powerful N-dimensional array object (`ndarray`).
- **Mathematical Functions**: Supports linear algebra, Fourier transforms, random number generation, and more.
- **Efficient Computation**: Performs operations efficiently due to the use of optimized C code under the hood.
- **Broadcasting**: Allows arithmetic operations on arrays of different shapes, making mathematical expressions concise.

### Basic Usage:

```python
import numpy as np

# Creating a 1D array
a = np.array([1, 2, 3, 4])

# Creating a 2D array (matrix)
b = np.array([[1, 2], [3, 4]])

# Basic operations
sum = np.sum(a)        # Sum of elements
mean = np.mean(b)      # Mean of the matrix
c = a + 5              # Broadcasting: add 5 to all elements of 'a'

print(a)
print(b)
```

### Useful Functions:

- `np.zeros(shape)`: Creates an array of zeros.
- `np.ones(shape)`: Creates an array of ones.
- `np.arange(start, stop, step)`: Creates an array with a range of values.
- `np.random.random(size)`: Generates random numbers in the given shape.
- `np.dot(a, b)`: Dot product of two arrays.
- `np.linalg.inv(a)`: Computes the inverse of a matrix.

### Array Manipulation:

- `np.reshape(a, newshape)`: Reshapes an array to the specified shape.
- `np.transpose(a)`: Returns the transpose of the array.
- `np.concatenate((a1, a2), axis=0)`: Concatenates two arrays along a given axis.

---

## 2. Pandas (Python Data Analysis Library)

Pandas is an essential library for data manipulation and analysis, built on top of NumPy. It provides two key data structures: `Series` (1D) and `DataFrame` (2D, tabular data).

### Key Features:

- **DataFrame**: A table-like structure with labeled axes (rows and columns).
- **Data Manipulation**: Supports filtering, grouping, joining, merging, and aggregation of data.
- **File I/O**: Easily reads and writes to various file formats (CSV, Excel, JSON, etc.).

### Basic Usage:

```python
import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Basic operations
mean_age = df['Age'].mean()   # Compute the mean of the 'Age' column
df['Age'] += 1                # Increment all values in 'Age' column by 1
df_filtered = df[df['Age'] > 30]  # Filter rows where 'Age' > 30

print(df)
```

### Useful Functions:

- `pd.read_csv(filepath)`: Reads a CSV file into a DataFrame.
- `df.head()`: Returns the first few rows of a DataFrame.
- `df.describe()`: Provides a summary of statistics for numerical columns.
- `df.groupby(column)`: Groups data by a column.
- `df.plot()`: Simple plotting of DataFrame data (using Matplotlib internally).

---

## 3. Matplotlib (Plotting Library)

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

### Key Features:

- **Wide Range of Plots**: Line plots, scatter plots, bar charts, histograms, and more.
- **Customization**: Extensive options for customizing plots (titles, labels, legends, colors).
- **Subplots**: Allows for multiple plots in a single figure.

### Basic Usage:

```python
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4]
y = [10, 20, 25, 40]

# Plotting a line graph
plt.plot(x, y, label='Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sample Plot')
plt.legend()
plt.show()
```

### Useful Functions:

- `plt.scatter(x, y)`: Creates a scatter plot.
- `plt.bar(x, height)`: Creates a bar chart.
- `plt.hist(data, bins)`: Creates a histogram.
- `plt.subplot(nrows, ncols, index)`: Creates subplots within a figure.
