# import tensorflow as tf
import numpy as np

# x1 = np.random.random_integers(5, size=(3,2))
x1 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]])
x2 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
print(x1.var(), x2.var())
w1 = np.random.randn(10, 10)
w2 = np.random.randn(10, 10)
w2[0][0] = 100
print(w1.var())

# f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
# x = np.random.randn(10, 1) # random input vector of three numbers (3x1)
# h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (4x1)
# h2 = f(np.dot(W2, h1) + b2) # calculate second hidden layer activations (4x1)
# out = np.dot(W3, h2) + b3

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
