import numpy as np
from matplotlib import pyplot as ppl
from scipy.interpolate import interp1d

def shenn(A, T, foo, t):
    def h(t, T):
        return np.sin(np.pi*t/T)/(np.pi*t)
    arg = np.arange(-A, A, T)
    F = foo(arg)
    Valh = h(t-arg, T)
    return T * np.sum(F*Valh)


A = 16
# T = np.arange(1/200, 1, 1/200)
T = 1/13
t = np.sqrt(2)
arg = np.arange(-A, A, T)
# F = np.sin(12*np.pi*arg)/(1+10*arg**4)
print(np.sin(2*np.pi*6*t))
print(shenn(A, T, foo=lambda x: np.sin(2*np.pi*6*x), t=t))
G = interp1d(arg, [np.sin(12*np.pi*x) for x in arg], kind='quadratic')
print(G(t))
# FF = np.fft.fft(F)
# ppl.plot(abs(FF))
# res = [shenn(A, y, foo=lambda x: np.sin(12*np.pi*x)/(1+10*t**4), t=t) for y in T]
ppl.plot(G)
ppl.show()