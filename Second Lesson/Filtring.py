import numpy as np
import matplotlib.pyplot as ppl
from scipy import signal as sgn

def f(t):
    return np.sin(300*t) + np.cos(400*t)

N = 2048
arg = np.arange(N)/N
F = f(arg[:int(N/2)])
FF = np.fft.fft(F)
m1 = np.amax(FF[:100])
ind1 = np.where(FF[:100]==m1)
print(ind1)
m2 = np.amax(FF[900:])
ind2 = np.where(FF[900:]==m2)
print(ind2)

ppl.plot(FF)
ppl.show()

# help(sgn.firwin)