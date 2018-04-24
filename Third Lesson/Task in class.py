import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sgn

N = 1024
w1 = 300  # чистый сигнал
w2 = 400  # шум
arg = np.arange(N)/N
S1 = np.sin(2*np.pi*w1*arg)
S2 = np.sin(2*np.pi*w2*arg)
Y = S1 + S2
# plt.plot(Y[:200])
# plt.show()

n = 11# Длина фильтра
T = 1/N
cutoff = 350 * 2 * T# Частота отсечения
# print(cutoff)
# B = sgn.firwin(n, cutoff)
B, A = sgn.butter(7, cutoff) # Фильтр Баттеруорта
u, v = sgn.freqz(B, 1) # Рисует фильтр
# plt.plot(u, abs(v))
# print(B)
Z = sgn.lfilter(B, A, Y)
plt.plot(S1[:50])
plt.plot(Z[:50])
plt.show()