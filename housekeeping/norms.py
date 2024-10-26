import numpy as np

array = np.array([3,3])
l1_norm = np.linalg.norm(array, 1)
l2_norm = np.linalg.norm(array)
var = np.var(array)
print(l1_norm, l2_norm, var)

array = np.array([6,0])
l1_norm = np.linalg.norm(array, 1)
l2_norm = np.linalg.norm(array)
var = np.var(array)
print(l1_norm, l2_norm, var)
