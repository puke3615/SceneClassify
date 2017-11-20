import numpy as np
import heapq

a = [
    [2, 1, 3],
    [4, 9, 5],
]
a = np.array(a)
b = a.argsort()[:, -2:][:, ::-1]
print(b)
