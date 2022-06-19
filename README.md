# Matrix exponential application of sparse matrix
This function computes $e^Ab$, where $A$ is a sparse matrix and $b$ a dense vector.

This is pretty much a copy paste of the scipy implementation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html)

## TODO
- Fix some CMake stuff
- Make header only
    - Now hard coded to double values

