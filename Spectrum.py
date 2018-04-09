import numpy as np
#from numpy,fft import fft
from scipy.io.wavfile import read
from matplotlib import pyplot as ppl
from scipy.fftpack import fft,ifft

'''
A = [1,2,3,4]
AF = np.fft.fft(A)
AA = np.fft.ifft(AF)

'''


N = 1000

arg = np.arange(N)/N
F = np.sin(30*(np.log10(arg+1)*arg))
F1 = F[:750]
F_F = np.fft.fft(F)
FF1 = np.fft.fft(F1)
m = np.amax(abs(FF1)[:6])
print(m)
i = np.where(abs(FF1)[:6]==m)
print(i)
ppl.plot(abs(FF1))
ppl.show()
#Orig = np.real(ifft(F_F))
