from numpy import *
from scipy.fftpack import fft,ifft
from scipy.io.wavfile import read,write
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
import os


# wnd = signal.hann(frameLen)
# frame *=wnd

def noiseFFT(no,frameLen):
    wnd = signal.hamming(frameLen)
    # frame *=wnd
    Flen = frameLen//2
    Ln = len(no)
    sqres = zeros(Flen)
    mns = zeros(Flen)
    K=0
    for Ind in range(0,Ln-frameLen,Flen//2):
        Span = no[Ind:Ind+frameLen]*wnd
        FSpan = abs(fft(Span))[0:Flen]
        sqres += (FSpan*FSpan)
        mns += FSpan
        K +=1
    sqres /= K
    mns /= K
    return sqres - mns*mns, K


def restSign(sign,frameLen,SigNo,parm=1):

    # frame *= wnd
    Ln = len(sign)
    Flen = frameLen//2
    wnd = signal.hann(frameLen)
    U = wnd[frameLen//2:]+wnd[:frameLen//2]
    Out = zeros((Ln))
    Four = zeros((Ln))
    for Ind in range(0,Ln - frameLen,Flen):
        Span = np.float32(sign[Ind:Ind + frameLen])*wnd
        FSpan = fft(Span)
        AFSpan = abs(FSpan)[0:Flen]
        SqF = AFSpan * AFSpan - parm * SigNo
        NFour = complex_(zeros(frameLen))
        for LInd in range(Flen):
            if SqF[LInd] > 0:
                NFour[LInd] = sqrt(SqF[LInd])
                NFour[LInd] *= FSpan[LInd]/AFSpan[LInd]
        for LInd in range(1,Flen):
            NFour[frameLen - LInd] = conj(NFour[LInd])
        NFour[0] = 0
        #plt.plot(abs(NFour))
        Four[Ind:Ind + frameLen] = NFour
        INFour = ifft(NFour)
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

def union_frames():
    import wave
    import os
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
    output.writeframes(data[0][1])
    output.writeframes(data[1][1])
    output.close()

# union_frames()
[fr,dt]=read('L16No.wav')
print(len(dt))
frameLen = 2048 #int(2*round(fr/200))
print(frameLen)
files = os.listdir('/Users/sadcihkova/Documents/Stolov/Task1/noise/')
dev_res = np.zeros((frameLen//2))
res_n = 0
[fr1,no]=read('noise.wav')
#plt.plot(abs(fft(no)))
for file in files:
    [fr1,no]=read('/Users/sadcihkova/Documents/Stolov/Task1/noise/'+file) #'noise.wav')
    dev, n = noiseFFT(no, frameLen)
    dev_res+=dev*n
    res_n+=n
dev_res/=res_n
# plt.plot(abs(fft(no)))
# frameLen1 = int(2*round(fr1/200))
# print(frameLen1)

dir = 'result'
if not os.path.exists(dir):
    os.makedirs(dir)
gama=27
out = restSign(dt, frameLen, dev_res, gama)
print(max(out))
write(dir+'/result_gama_%d_len_%d.wav'%(gama, frameLen), fr,np.asarray(out, dtype=np.int16))

