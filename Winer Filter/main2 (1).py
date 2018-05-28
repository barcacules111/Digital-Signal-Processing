import numpy as np
from scipy import signal as sgn
from numpy import *
from scipy.fftpack import fft,ifft
from scipy.io.wavfile import read,write
from scipy import signal
from matplotlib import pyplot as plt
import os
import wave

def union_frames():
    dir = 'noise/'
    infiles = [os.path.join(dir, el) for el in os.listdir(dir)]
    #infiles = ["sound_1.wav", "sound_2.wav"]
    outfile = "noise.wav"

    data = []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()

    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()

# union_frames()
frameLen = 2048
[Fr1, Dat] = read('LL17.wav')
N = len(Dat)
[fr1,DataNoise]=read('noise.wav')
result = np.array(DataNoise, dtype=np.float64)
while len(result)<N:
    result = np.concatenate((result, DataNoise), 0)
result_noise = result[:N]

# def akf(noise_signal, frameLen):
#     Ln = len(noise_signal)
#     Flen = frameLen//2
#     Rn = np.zeros((frameLen))
#     K = 0
#     for Ind in range(0,Ln - 2*frameLen,Flen):
#         NoiseSpan = np.float32(noise_signal[Ind+Flen:Ind +frameLen+Flen])
#         NoiseSpan_ = np.float32(noise_signal[Ind:Ind + 2*frameLen-1])
#         Rn = Rn+np.correlate(NoiseSpan_, NoiseSpan, 'valid')
#         # Rn = getFFT(NoiseSpan)
#         K+=1
#     return Rn/K

# def getFFT(Span):
#     wnd = signal.hamming(frameLen)
#     Span = Span#*wnd
#     FSpan = fft(Span)
#     return abs(FSpan)**2

def acf(noise_signal, frameLen, max_val):
    Ln = len(noise_signal)
    Flen = frameLen//2
    Rn = np.zeros((frameLen))
    K = 0
    hann1 = sgn.hann(frameLen)
    hann2 = sgn.hann(2*frameLen-1)
    for Ind in range(Flen,Ln - 2*frameLen,Flen):
        NoiseSpan = np.float32(noise_signal[Ind:Ind +frameLen])#*hann1
        NoiseSpan_ = np.float32(noise_signal[Ind-Flen:Ind + frameLen+Flen-1])#*hann2
        Rn = Rn+np.correlate(NoiseSpan_/max_val, NoiseSpan/max_val, 'valid')
        # Rn = getFFT(NoiseSpan)
        K+=1
    return Rn/K

def restSign_scipy(sign,frameLen,parm=1):
    Ln = len(sign)
    Flen = frameLen//2
    wnd = signal.hann(frameLen)
    wnd2 = signal.hann(2047)
    U = wnd[frameLen//2:]+wnd[:frameLen//2]
    Out = zeros((Ln))
    Four = zeros((Ln))
    a = 0.5
    for Ind in range(0,Ln - frameLen,Flen):
        Span = np.float32(sign[Ind:Ind +frameLen])*wnd
        result = sgn.wiener(Span)
        Out[Ind + Flen:Ind + frameLen] = result[Flen:]
        if Ind != 0:
            Out[Ind:Ind + Flen] = (result[:Flen]+Out[Ind:Ind + Flen])/U # custom_hamm(Out[Ind:Ind + Flen], INFour[:Flen], frameLen)
        else:
            Out[Ind:Ind + Flen] = result[:Flen]
    return Out


def restSign(sign,frameLen, noise_signal,parm=1):
    Ln = len(sign)
    Flen = frameLen//2
    wnd = signal.hamming(frameLen)
    wnd2 = signal.hann(2047)
    U = wnd[frameLen//2:]+wnd[:frameLen//2]
    Out = zeros((Ln))
    Four = zeros((Ln))
    a = 0.5
    # noise_signal_ = np.concatenate((noise_signal[:Flen],noise_signal, noise_signal))
    sign_ = np.concatenate((sign[:Flen],sign, sign[-Flen:]))
    max_val = max(sign)
    mean_val = mean(sign)
    Rn = acf(noise_signal-mean_val, frameLen, max_val)*wnd*0.5#/1024
    plt.plot(Rn)
    plt.cla()
    hann2 = sgn.hann(2*frameLen-1)
    for Ind in range(0,Ln - frameLen,Flen):
        Span = np.float32(sign[Ind:Ind +frameLen])#*wnd
        Span_ = np.float32(sign_[Ind:Ind + 2*frameLen-1])#*hann2
        # tmp_Span = np.concatenate((Span[:Flen-1],Span, Span[Flen:]), axis=0)*wnd2
        # Span*=wnd

        # NoiseSpan = np.float32(sign[Ind:Ind + frameLen])
        # tmp_NoiseSpan = np.concatenate((NoiseSpan[:Flen - 1], NoiseSpan, NoiseSpan[Flen:]), axis=0) * wnd2
        # NoiseSpan *= wnd


        # NoiseSpan = (noise_signal[Ind:Ind+frameLen])
        # NoiseSpan_ = (noise_signal_[Ind:Ind+2*frameLen-1])
        #NoiseSpan = NoiseSpan - mean(NoiseSpan)
        # ynorm1 = np.sum(NoiseSpan**2)
        #Span2 = Span - mean(Span)
        # ynorm2 = np.sum(Span ** 2)
        # Ry = getFFT(Span)#/ynorm2
        Ry = np.correlate((Span_-mean(Span))/max_val, (Span-mean(Span))/max_val, 'valid')*wnd*0.5#/1024
        k=1
        # for i in range(-5, 5):
        #     try:
        #         Span = np.float32(sign[Ind+i:Ind + frameLen+i])
        #         Span_ = np.float32(sign_[Ind+i:Ind+i + 2 * frameLen - 1])
        #         Ry += np.correlate((Span_-mean(Span))/max_val, (Span-mean(Span))/max_val, 'valid')/1024
        #         k+=1
        #     except:
        #         pass
        Ry/=k
        plt.plot(Ry)
        plt.cla()
        # Rn = np.correlate(NoiseSpan_/2**15, NoiseSpan/2**15, 'valid')#/ynorm1
        # plt.plot(Rn)
        # plt.cla()
        if Ry.shape != Rn.shape:
            print(Ind, Ln, Ry.shape, Rn.shape)
        Rx = Ry - Rn
        plt.plot(Rx)
        plt.cla()
        RFy = fft(Ry)
        RFx = fft(Rx)
        H = RFx / RFy
        plt.plot(abs(H))
        plt.cla()
        # h = ifft(H)
        # h = abs(h)
        # Span = np.float32(sign[Ind:Ind + frameLen]) * wnd
        # INFour = sgn.lfilter(h, [1], Span)
        # fft_out = fft(Dat) * H
        # out = ifft(fft_out)
        FSpan = fft(Span*wnd)
        plt.plot(FSpan)
        plt.cla()
        AFSpan = abs(FSpan)
        NFour = AFSpan*abs(H)
        for LInd in range(frameLen):
            NFour[LInd] *= FSpan[LInd]/AFSpan[LInd]
        plt.plot(abs(NFour))
        plt.cla()
        for LInd in range(1,Flen):
            NFour[frameLen - LInd] = conj(NFour[LInd])
        NFour[0] = 0
        # Four[Ind:Ind + frameLen] = NFour
        # INFour =
        INFour = real(ifft(NFour))
        # result.append(INFour) #*wnd
        Out[Ind+Flen:Ind + frameLen] = INFour[Flen:]
        if Ind != 0:
            Out[Ind:Ind + Flen] = (INFour[:Flen]+Out[Ind:Ind + Flen])/U # custom_hamm(Out[Ind:Ind + Flen], INFour[:Flen], frameLen)
        else:
            Out[Ind:Ind + Flen] = INFour[:Flen]
    #hamming_window_out = Hamming_window(np.array(result))
    plt.figure()
    plt.plot(Out)
    plt.show()
    plt.figure()
    #plt.plot(abs(fft(Out[35000:65000])))
    #plt.plot(abs(Four))
    # for Ind in range(frameLen,Ln - frameLen,frameLen):
    #     Out[Ind-Flen//2:Ind+Flen//2] = Out[Ind-Flen//2:Ind+Flen//2]*wnd
    return real(Out)
dir = 'result'
cutOff = 1000 # Cutoff frequency
nyq = 0.5 * Fr1
# cutOff1 = 1450
# cutOff2 = 1550
# fc1 = cutOff1 / nyq
# fc2 = cutOff2 / nyq
# print(fc1, fc2)
cutOff=150
fc = cutOff/nyq
# prin t(fc)
B, A = sgn.butter(9, 0.2)
U, V = sgn.freqz(B, A)# рисует передаточную функцию фильтра
plt.plot(U, abs(V))
out = restSign(np.array(Dat, dtype=np.float64), frameLen, result)
fout = fft(out[:1024])
plt.cla()
plt.figure()
plt.plot(fout)
plt.show()
out = sgn.lfilter(B, A, out)
write('result4.wav', Fr1, np.asarray(out, dtype=np.int16))