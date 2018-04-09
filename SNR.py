import numpy as np
import matplotlib.pyplot as ppl

arg = np.arange(1000)/1000
F = np.sin(30*arg) + np.sin(70*arg)
B = 15 # Количество бит для оцифровки
delta = (np.amax(F) - np.amin(F))/2**B
F1 = F - np.amin(F)
F1 /= delta
F1 = np.around(F1) * delta
print(F1)
Err_var = np.var(F1-F)
F_var = np.var(F)
print(Err_var)
print(F_var)
SNR = 10*np.log10(F_var/Err_var)
print(SNR)