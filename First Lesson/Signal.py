import numpy as np
import matplotlib.pyplot as ppl


arg = np.arange(1000)/1000
F = np.sin(30*arg) + np.sin(70*arg)
B = 15 # bit numbers
delta = (np.amax(F) - np.amin(F))/2**B
F1 = F - np.amin(F)
F1 /= delta
F1 = np.around(F1) * delta

print(F1)

F -= np.amin(F)
Error_var = np.var(F1-F)
F_var = np.var(F)

print(Error_var)

SNR = 10*np.log10(F_var/Error_var)

print(SNR)




''' for i in range(num_v): B = numpy.copy(A) A = numpy.minimum(B, B[:,k] + B[k,:])
return A


def c_inf_comp(z):
    if z>0: return np.exp(-1./(z*z))
    else:
        return 0
x = np.array([-10., 10.]) >>> x array([-10., 10.])
c_inf_comp(x[0]) 0 >>>
c_inf_comp(x[1]) 0.99004983374916811
vfunz = np.vectorize(c_inf_comp)
vfunz(x) array([0, 0])


print()'''

