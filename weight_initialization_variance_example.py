import numpy as np

print("=== pre processing ===")
# input x
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
print("input x: %f" % x.var())

# high variance w1
w1 = np.random.randn(10, 10)
w1[0][0] = 100
print("parameter w1 high variance : %f" % w1.var())

# low variance w2
w2 = np.random.randn(10, 10)
print("parameter w2 low variance : %f" % w2.var())

print("")
print("=== 1 layer nn feed forward ===")
# output1 with high variance input w1
h1 = np.dot(w1, x)
print("output h1 by w1 : %f" % h1.var())

# output2 with high variance input w2
h2 = np.dot(w2, x)
print("output h2 by w2 : %f" % h2.var())

print("")
print("=== 3 layers nn feed forward ===")
h3 = np.dot(w1, np.dot(w1, np.dot(w1, x)))
print("output h3 by w1 : %f" % h3.var())

h4 = np.dot(w2, np.dot(w2, np.dot(w2, x)))
print("output h4 by w2 : %f" % h4.var())

'''
=== pre processing ===
input x: 8.250000
parameter w1 high variance : 100.044956
parameter w2 low variance : 0.901882

=== 1 layer nn feed forward ===
output h1 by w1 : 2006.573013
output h2 by w2 : 366.864624

=== 3 layers nn feed forward ===
output h3 by w1 : 140190058599.812500
output h4 by w2 : 23940.654573
'''
