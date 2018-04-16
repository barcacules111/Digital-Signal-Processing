import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
from scipy.io.wavfile import read, write

Fr, file = read('NarrowNo.wav')
print(len(file))
FF = np.fft.fft(file[:2048])
m1 = np.amax(abs(FF[:1024]))
print(m1)
ind1 = np.where(abs(FF[:1024])==m1)[0]
print(ind1)
m2 = np.amax(abs(FF[100:1024]))
print(m2)
ind2 = np.where(abs(FF[100:1024])==m2)[0] + 100
print(ind2)
fr1 = ind1 * Fr / 2048
fr2 = ind2 * Fr / 2048
print(fr1)
print(fr2)

# plt.plot(FF)
# plt.show()

n = 101# Длина фильтра
cutoff = (2*1200/Fr,2*1210/Fr)# Частота отсечения
# print(cutoff)
B = sgn.firwin(n, cutoff)
# B, A = sgn.butter(7, cutoff, 'bandstop') # Фильтр Баттеруорта
u, v = sgn.freqz(B, 1) # Рисует фильтр
plt.plot(u, abs(v))
# print(B)
Z = sgn.lfilter(B, 1, file)
write('1.wav', Fr, Z.astype(np.int16))
# plt.show()