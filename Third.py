import numpy as np
import matplotlib.pyplot as ppl

N = 1000
arg = np.arange(N)/N
F = np.sin(30*np.sqrt(arg+1)*arg)
F1 = F[250:]
FF = np.fft.fft(F)
FF1 = np.fft.fft(F1)
m = np.amax(abs(FF1)[:6])
print(m)
ind = np.where(abs(FF1)[:6]==m)
print(ind)
# ppl.plot(abs(FF))
# ppl.show()
# A = [1,2,3,4]
# AF = np.fft.fft(A)
# AA = np.real(np.fft.ifft(AF))
# print(AF)
# print(AA)