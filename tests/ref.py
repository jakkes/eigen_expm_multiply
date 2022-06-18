import numpy as np
from scipy.linalg import expm


A = np.random.randn(4, 4)
b = np.random.randn(4, 1)


print(A)
print(b)
print(expm(A) @ b)
