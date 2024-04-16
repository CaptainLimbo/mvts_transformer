import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

a = np.array([1, 2])
b = np.array([3, 4])
print(np.bincount(b).argmax())
