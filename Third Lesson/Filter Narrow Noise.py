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

n = 505  # Длина фильтра
cutoff = (fr1[0]-100,fr1[0]+100)# Частота отсечения
# print(cutoff)
B = sgn.firwin(n, cutoff, fs=Fr)
# B, A = sgn.butter(7, cutoff, 'bandstop') # Фильтр Баттеруорта
u, v = sgn.freqz(B, 1) # Рисует фильтр
# plt.plot(u, abs(v))
# print(B)
Z = sgn.lfilter(B, 1, file)
FF1 = np.fft.fft(Z[:2048])
plt.plot(FF1)
write('1.wav', Fr, Z.astype(np.int16))
plt.show()

cutoff = (fr2[0]-100, fr2[0]+100)
B = sgn.firwin(n, cutoff, fs=Fr)
u, v = sgn.freqz(B, 1) # Рисует фильтр
# plt.plot(u, abs(v))
Z2 = sgn.lfilter(B, 1, Z)
FF2 = np.fft.fft(Z2[:2048])
plt.plot(FF2)
write('2.wav', Fr, Z2.astype(np.int16))
plt.show()

# plt.plot(file[:200])
# plt.plot(Z[:200])
# plt.plot(Z2[:200])
# plt.show()