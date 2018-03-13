import numpy as np

print("=== pre processing ===")
# high variance input x1
x1 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]])
# low variance input x2
x2 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
print("input x1 high variance : %f" % x1.var())
print("input x2 low variance : %f" % x2.var())

# gaussian distribution weights w
w = np.random.randn(10, 10)
print("parameter w gaussian distribution : %f" % w.var())

print("")
print("=== 1 layer nn feed forward ===")
# output1 with high variance input x1
h1 = np.dot(w, x1)
print("output h1 by x1 : %f" % h1.var())

# output2 with high variance input x2
h2 = np.dot(w, x2)
print("output h2 by x2 : %f" % h2.var())

print("")
print("=== 3 layers nn feed forward ===")
h3 = np.dot(w, np.dot(w, np.dot(w, x1)))
print("output h3 by x1 : %f" % h3.var())

h4 = np.dot(w, np.dot(w, np.dot(w, x2)))
print("output h4 by x2 : %f" % h4.var())

'''
=== pre processing ===
input x1 high variance : 818.250000
input x2 low variance : 8.250000
parameter w gaussian distribution : 0.962264

=== 1 layer nn feed forward ===
output h1 by x1 : 2260.361338
output h2 by x2 : 110.130665

=== 3 layers nn feed forward ===
output h3 by x1 : 379973.211796
output h4 by x2 : 4379.333152
'''
