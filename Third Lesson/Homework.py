from numpy import *
from scipy.fftpack import fft,ifft
from scipy.io.wavfile import read,write
import matplotlib.pyplot as ppl
import numpy as np
from scipy import signal as sgn
from scipy.signal import firwin

def noiseFFT(dnoise, Flen):
    FL = Flen//2
    Ln = len(dnoise)
    sqres = np.zeros(FL)
    mns = np.zeros(FL)
    K = 0
    for Ind in range(0,Ln-FL+1,FL):
        Span = dnoise[Ind:Ind+FL]
        FSpan = abs(np.fft.fft(Span))[0:FL]
        sqres += (FSpan*FSpan)
        mns += FSpan
        K = K + 1
    sqres /= K
    mns /= K
    return sqres - mns*mns

rate, data = read("L14No.wav")
rate1, data2=read("noise.wav")
size=2
dataemb = np.split(data2, 2, 1)[0]
i=0
noise=[]
while i<len(data2):
    noise.append(dataemb[i][0])
    i+=1
disp=noiseFFT(noise[:],rate//100*size)
rate2, data3=read("noise2.wav")

dataemb = np.split(data3, 2, 1)[0]
i=0
noise=[]
while i<len(data3):
    noise.append(dataemb[i][0])
    i+=1
disp1=noiseFFT(noise[:],rate//100*size)
disp+=disp1
disp/=2
print(len(disp))
ppl.plot(data)

def restSign(sign, frameLen, SigNo, parm,lowkey,rate):
    Ln = len(sign)
    padded_signal=zeros(frameLen+(math.ceil(Ln/frameLen))*frameLen)
    padded_signal[frameLen//2:frameLen//2+Ln]=sign
    Flen = frameLen // 2
    Out = zeros(len(padded_signal))
    for Ind in range(0, len(padded_signal) - frameLen//2, frameLen//2):
        Span = padded_signal[Ind:Ind +frameLen]
        Span=complex_(Span)
        b = firwin(11, [500 / rate, 7000 / rate], False)
        Span = sgn.lfilter(b, 1, INFour)
        #wnd = hanning(frameLen)
        wnd=hamming(frameLen)
        Span*=wnd
        FSpan = fft(Span)
        AFSpan = abs(FSpan)[0:Flen]
        SqF = AFSpan * AFSpan - parm * SigNo
        flr = lowkey * AFSpan * AFSpan
        NFour = complex_(zeros(frameLen))
        for LInd in range(Flen):
            if SqF[LInd] > flr[LInd]:
                NFour[LInd] = sqrt(SqF[LInd])
            else:
                NFour[LInd] = sqrt(flr[LInd])
            NFour[LInd] *= FSpan[LInd]/AFSpan[LInd]
        for LInd in range(1, Flen):
            NFour[frameLen - LInd] = conj(NFour[LInd])
        NFour[0] = 0
        INFour = ifft(NFour)
        Out[Ind:Ind + frameLen] += real(INFour)
        for i in range(0,frameLen//2):
            Out[i+Ind]/=(wnd[i]+wnd[i+frameLen//2])
    return real(Out[frameLen//2:frameLen//2+Ln])


res=restSign(data,rate//100*size,disp,21,0.001,rate)

ppl.plot(res)
write('res.wav',rate,res.astype(np.int16))
# z=max(abs(res))
# res/=z
# res*=2**15
# # ppl.plot(res)
# ppl.show()
# write('res_norm.wav',rate,res.astype(np.int16))