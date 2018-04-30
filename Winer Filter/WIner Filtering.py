from numpy import *
from scipy.fftpack import fft,ifft
from scipy.io.wavfile import read,write
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
import os


def sgnFFT(sgn,frameLen):
    wnd = signal.hamming(frameLen)
    # frame *=wnd
    Flen = frameLen//2
    Ln = len(sgn)
    mns = complex_(zeros(Flen))
    K=0
    for Ind in range(0,Ln-frameLen,Flen//2):
        Span = sgn[Ind:Ind+frameLen]*wnd
        FSpan = fft(Span)[0:Flen]
        mns += FSpan
        K += 1
    mns /= K
    return mns, K


def filter_fft(noise_fft, signal_fft):
    return 1 - noise_fft/signal_fft


def restSign(sign, frameLen, noise_fft):
    # frame *= wnd
    Ln = len(sign)
    Flen = frameLen//2
    wnd = signal.hann(frameLen)
    U = wnd[frameLen//2:]+wnd[:frameLen//2]
    Out = zeros((Ln))
    Four = complex_(zeros((Ln)))
    # signal_fft, _ = sgnFFT(sign, frameLen)
    # h_fft = filter_fft(noise_fft, signal_fft)

    for Ind in range(0,Ln - frameLen,Flen):
        Span = np.float32(sign[Ind:Ind + frameLen])*wnd
        FSpan = fft(Span)
        NFour = complex_(zeros(frameLen))
        NFour[:Flen] = FSpan[:Flen]*filter_fft(noise_fft, FSpan[:Flen])
        for LInd in range(1,Flen):
            NFour[frameLen - LInd] = conj(NFour[LInd])
        NFour[0] = 0
        print('FSpan = ', FSpan[:Flen])
        print('H = ', filter_fft(noise_fft, FSpan[:Flen]))
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
# frameLen = 2048
frameLen = int(2*round(Fr/50))
files = os.listdir('C:\\Users\\Илюха\\Documents\\GitHub\\Digital-Signal-Processing\\Winer Filter\\Noise')
print(files[:2])
noise_fft_res = complex_(np.zeros((frameLen // 2)))
res_n = 0
for f in files[:2]:
    fr1, no = read('Noise\\' + f)
    dev, n = sgnFFT(no, frameLen)
    noise_fft_res += dev * n
    res_n += n
noise_fft_res /= res_n

out = restSign(file, frameLen, noise_fft_res)
print(out)
write('Result.wav', Fr, np.asarray(out, dtype=np.int16))
