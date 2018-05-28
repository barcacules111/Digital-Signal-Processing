from numpy import *
from scipy import signal as sgn
from scipy.fftpack import fft,ifft
from scipy.io.wavfile import read,write
from scipy.signal import firwin
from matplotlib import pyplot as plt
import numpy as np
import os


def AutoCorrelation(x):
    x = np.float32(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result


def noise_ac(noise_signal, frameLen):
    Ln = len(noise_signal)
    Flen = frameLen//2
    Rn = zeros(frameLen)
    for ind in range(0,Ln - frameLen,Flen):
        Rn += AutoCorrelation(noise_signal[ind:ind+frameLen])
    Rn /= (Ln - frameLen)//Flen
    return Rn


def restSign(sign, frameLen, Rn):
    # frame *= wnd
    Ln = len(sign)
    Flen = frameLen//2
    wnd = sgn.hamming(frameLen)
    U = wnd[frameLen//2:]+wnd[:frameLen//2]
    Out = zeros((Ln))
    Four = complex_(zeros((Ln)))
    for Ind in range(0,Ln - frameLen,Flen):
        Span = np.float32(sign[Ind:Ind + frameLen])
        FSpan = fft(Span)
        Ry = AutoCorrelation(Span)
        Rx = Ry - Rn
        RFy = fft(Ry)
        RFx = fft(Rx)
        H = RFx/RFy
        NFour = complex_(zeros(frameLen))
        NFour[:Flen] = FSpan[:Flen]*H[:Flen]
        for LInd in range(1,Flen):
            NFour[frameLen - LInd] = conj(NFour[LInd])
        NFour[0] = 0
        print('FSpan = ', FSpan[:Flen])
        print('H = ', H)
        print('Res = ', NFour[:Flen])

        #plt.plot(abs(NFour))
        Four[Ind:Ind + frameLen] = NFour
        INFour = real(ifft(NFour))
        # result.append(INFour) #*wnd
        Out[Ind+Flen:Ind + frameLen] = INFour[Flen:]
        if Ind != 0:
            Out[Ind:Ind + Flen] = (INFour[:Flen]+Out[Ind:Ind + Flen])/U
        else:
            Out[Ind:Ind + Flen] = INFour[:Flen]
    plt.figure()
    plt.plot(Out)
    plt.figure()
    return real(Out)


Fr, file = read('LL17.wav')

# print(len(file))
# write('Noise/Noise1.wav', Fr, file[round(Fr*0.55):round(Fr*1)])
# write('Noise/Noise3.wav', Fr, file[round(Fr*6):round(Fr*6.9)])

Frn, filen = read('noise.wav')

frameLen = 2048
Rn = noise_ac(filen,frameLen)
out = restSign(file, frameLen, Rn)
print(out)
B, A = sgn.butter(9, 0.2)
out = sgn.lfilter(B, A, out)
# b = firwin(121, 3000/Fr, pass_zero=True)
# out = sgn.lfilter(b, 1, out)
write('Result3.wav', Fr, np.asarray(out, dtype=np.int16))