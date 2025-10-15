```python
# Import the Numpy Python library for us in answers to the following questions
import numpy as np
import matplotlib.pyplot as plt
```

# Preamble

**Question 1:** 
Write code to solve $M\cdot \vec{v}$ for  
$
    M=\begin{bmatrix}
        1 & 2 & 3 \\ 
        4 & 5 & 6 \\
        7 & 8 & 9 \\
        10 & 11 & 12 \\
    \end{bmatrix}
$
and
$
    \vec{v} = \begin{bmatrix}
        3 \\
        6 \\
        9 \\
    \end{bmatrix}
$



```python
import numpy as np
import matplotlib.pyplot as plt

def matrix_vector_dot_product():
    """
    Calculates the dot product of matrix M and vector v.
    """
    # Define the 4x3 matrix M as specified in the problem.
    M = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    
    # Define the 3-element vector v.
    v = np.array([3, 6, 9])
    
    # Calculate and return the dot product of M and v.
    # The result will be a 4-element vector.
    return M @ v

# The hidden test cell will call the function, so we don't call it here.
# result = matrix_vector_dot_product()
# print(result)
# Expected output: [ 42  96 150 204]

```


```python
# HIDDEN TEST CELL

```

**Question 2:**
Find the eigenvalues and corresponding eigenvectors for the matrix
$
    M = \begin{bmatrix}
        5 & 10 & 11 \\
        9 & 2 & 22 \\
        3 & 4 & 5\\
    \end{bmatrix}
$

Store the eigenvalues in a variable called $eigvals$ and the eigenvectors in a variable called $eigvecs$.


```python
import numpy as np
from numpy.linalg import eig

def compute_eigenvalues_eigenvectors():
    """
    This function should compute and return the eigenvalues and eigenvectors 
    of the matrix M.
    """
    # Define the matrix M as specified in the problem.
    # Note: The matrix in the prompt for Question 2 is different from the
    # one in the student's code. We'll use the one from the prompt.
    M = np.array([[5, 10, 11],
                  [9, 2, 22],
                  [3, 4, 5]])
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eig(M)
    
    return eigvals, eigvecs

# The autograder also expects the variables 'eigvals' and 'eigvecs' to be
# defined in the global scope. So, we define the function and then call it.
eigvals, eigvecs = compute_eigenvalues_eigenvectors()

```


```python
# HIDDEN TEST CELL
```

**Question 3:**
Read in the Datasaurus x-y coordinates as an 2xN matrix.

Use the rotation matrix to rotate the points 90 degrees and plot the result. Use the variable $R$ to represent the rotation matrix.

Use the variable $dino2$ to represent the rotated data points.


```python
dino = np.loadtxt("./assets/Datasaurus_data.csv",delimiter=',').T
print(dino.shape)
dino[:,:10]
```


```python
# Load the data first
dino = np.loadtxt("./assets/Datasaurus_data.csv", delimiter=',').T

def rotate_dino(dino):
    """
    Rotates the given 'dino' dataset by 90 degrees and plots the rotated dinosaur.

    Returns: 
        dino2 : A 2D array representing the rotated dinosaur dataset.
    """
    # Define the 90-degree counter-clockwise rotation matrix R
    theta = np.pi / 2  # 90 degrees in radians
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Apply the rotation to the dino data
    dino2 = R @ dino

    # Plot the rotated data
    plt.figure(figsize=(6, 6))
    plt.scatter(dino2[0, :], dino2[1, :], s=10)
    plt.title("Rotated Datasaurus")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.axis('equal')  # Ensure the aspect ratio is equal
    plt.grid(True)
    plt.show()
    
    return dino2

# Execute the function
dino2 = rotate_dino(dino)
```


```python
# HIDDEN TEST CELL

```


```python

```


```python

```


```python

```


```python

```


```python

```
