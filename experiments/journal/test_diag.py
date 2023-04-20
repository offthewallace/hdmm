import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

# create a 3x3 _ProductLinearOperator object
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
eigenvalues, eigenvectors = np.linalg.eig(A)

# sort eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# create diagonal matrix
D = np.diag(eigenvalues)

# extract diagonal
diag = np.diagonal(D)

print(diag)
