import numpy as np

# high variance input data x1
x1 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]])
# low variance input data x2
x2 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
print("x1 high variance : %f" % x1.var())
print("x2 low variance : %f" % x2.var())

# gaussian distribution weights w
w1 = np.random.randn(10, 10)
w1[0][0] = 100
# log variance weights w2
w2 = np.random.randn(10, 10)
print("w1 high variance : %f" % w1.var())
print("w2 low variance : %f" % w2.var())

ret = np.dot(w1, x1)
print(ret.var())
print(ret.mean())

ret = np.matmul(w1, x2)

print(ret.var())
print(ret.mean())

ret = np.matmul(w1, np.matmul(w1, np.matmul(w1, x1)))

print(ret.var())
print(ret.mean())

ret = np.matmul(w1, np.matmul(w1, np.matmul(w1, x2)))

print(ret.var())
print(ret.mean())

ret = np.matmul(w2, np.matmul(w2, np.matmul(w2, x2)))

print(ret.var())
print(ret.mean())
