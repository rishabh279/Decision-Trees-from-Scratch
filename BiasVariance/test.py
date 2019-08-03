import numpy as np

test = np.zeros((2, 3, 4))
print(test)
print(test[:, 0 , :])
test[:, 0 , 0] = 4
print(test)