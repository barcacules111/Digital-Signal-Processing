import numpy as np
from matplotlib import pyplot as ppl

# A=np.int_([1,2,3])
# B=A[2:]
# B[0]=-1
# print(A)
N = 50
A = 100 + np.pi/1000
T = 0.01
def f(x):
    return x**8/(1+x*x)
arg = np.arange(-A, A, T)
arr1 = f(arg)
arr2 = np.sin(2*np.pi*arg*N)/(np.pi*arg)
res = T * np.sum(arr1*arr2)
print(res)

ind = np.where(arg<1)
ind2 = np.where(arg<-1)
print(ind)
print(ind2)
ppl.plot(arr2[9900:10100])
ppl.show()