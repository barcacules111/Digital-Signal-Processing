import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
from scipy.io.wavfile import read, write

Fr, file = read('L17No.wav')
file = np.append(file, file[-1])  # Раз уж нам нужно четное N, то придётся добавиь элемент
print(Fr)
print(len(file))  # 595800 = 	2**3 ∙ 3**2 ∙ 5**2 ∙ 331
N = 331*4  # Получится 450 фрагментов (900 с половинным наложением)
coef = 5
frameLen = int(2*round(Fr/200))


def noiseFFT(no, frameLen):
    Flen = int(frameLen/2)
    Ln = len(no)
    sqres = np.zeros(Flen)
    mns = np.zeros(Flen)
    K = 0
    for Ind in np.arange(0,Ln-frameLen,frameLen):
        Span = no[Ind:Ind+frameLen]
        FSpan = abs(np.fft.fft(Span))[0:Flen]
        sqres += (FSpan*FSpan)
        mns += FSpan
        K +=1
    sqres /= K
    mns /= K
    return sqres - mns*mns


def restSign(sign, frameLen, SigNo, parm):
    Ln = len(sign)
    Flen = int(frameLen/2)
    Out = np.zeros(Ln)
    for Ind in np.arange(0,Ln - frameLen,frameLen):
        Span = sign[Ind:Ind + frameLen]
        FSpan = np.fft.fft(Span)
        AFSpan = np.abs(FSpan)[0:Flen]
        SqF = AFSpan * AFSpan - parm * SigNo
        NFour = np.complex_(np.zeros(frameLen))
        for LInd in np.arange(Flen):
            if SqF[LInd] > 0:
                NFour[LInd] = np.sqrt(SqF[LInd])
                NFour[LInd] *= FSpan[LInd]/AFSpan[LInd]
        for LInd in np.arange(1,Flen):
            NFour[frameLen - LInd] = np.conj(NFour[LInd])
        NFour[0] = 0
        INFour = np.fft.ifft(NFour)
        Out[Ind:Ind + frameLen] = np.real(INFour)
    return Out
# write('Noise.wav', Fr, file[round(Fr*3.3):round(Fr*3.85)])

Fr2, noise = read('Noise.wav')
print(len(noise))
# plt.plot(noise)
# plt.show()

variances = noiseFFT(noise, frameLen)

pure_signal = restSign(file, frameLen, variances, coef)
# FF = np.fft.fft(file)
# # phase = FF / np.abs(FF)
#
# x_ff = np.maximum(np.abs(FF)**2 - coef * variance, np.zeros(len(FF)))
# print(x_ff)
# x_ff = np.sqrt(x_ff)
#
# pure_signal = np.real(np.fft.ifft(x_ff))
write('L17NoPure.wav', Fr, pure_signal.astype(np.int16))
plt.plot(file[:1000])
plt.plot(pure_signal[:1000])
plt.show()

'''
plt.plot(file[:N])
# plt.show()

x_res = []
for i in np.arange(0, len(file)/N - 1, 0.5):
    FF = np.fft.fft(file[int(N*i):int(N*(i+1))])
    sigma_n = np.var(FF)
    phase = FF/np.abs(FF)

    x_ff = np.maximum(np.abs(FF)**2 - coef * sigma_n, np.zeros(len(FF)))
    x_ff = np.sqrt(x_ff) * phase
    x_res.append(x_ff)

x_final = x_res[0][:int(N/2)]
print(len(x_res))
for i in range(len(x_res)-1):
    arg = np.arange(0, 1, 2/N)
    # res = []
    # for j in range(int(N / 2)):
    #     r = x_res[i][int(N/2+j)] * arg[int(N/2-1-j)] + x_res[i+1][j] * arg[j]
    #     res.append(r)
    res = [x_res[i][int(N/2+j)] * arg[int(N/2-1-j)] + x_res[i+1][j] * arg[j] for j in range(int(N/2))]
    x_final = np.append(x_final, res)

pure_signal = np.real(np.fft.ifft(x_final))
# write('L17NoPure.wav', Fr, pure_signal.astype(np.int16))
plt.plot(pure_signal[:N])
plt.show()'''