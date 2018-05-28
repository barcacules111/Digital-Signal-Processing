import numpy as np
import math
import scipy.signal as sgn
from scipy.signal import  firwin
from scipy.signal import argrelextrema
from scipy.io.wavfile import read,write

def pitch_autocor(sign,rate):
    X=np.zeros(len(sign)*2)
    b = firwin(1201,(800/rate))
    Z = sgn.lfilter(b, 1, sign)
    print(Z)
    for i in range(len(sign)):
        X[i]=Z[i]
    X_f=np.fft.fft(X)
    X_mod=abs(X_f)**2
    auto_cor=np.fft.ifft(X_mod)
    auto_cor=auto_cor[:len(sign)]
    Z=argrelextrema(auto_cor,np.greater)
    i=0
    pitch=0
    for j in range(1,len(Z[0])):
        dif=Z[0][j]-Z[0][j-1]
        pitch+=dif
        i+=1
    print(i,"   ",pitch)
    pitch/=i
    return math.ceil(pitch)

def stretch(In,Coe):
    LnFrame = len(In)
    LnOut =  int(LnFrame * Coe)
    Out = np.zeros(LnOut)
    for I in range(LnOut):
        Indx = int(I/Coe)
        Out[I] = In[Indx]
    return Out

def pitch_change(coef,pitch,sign):
    res=[]
    Len=len(sign)
    segm=math.ceil(Len/pitch)
    new_pitch=math.ceil(pitch*coef)
    leftover=(Len-segm*new_pitch)//new_pitch
    distance=segm/leftover
    print(distance)
    K=0.0
    for Ind in range(0,Len,pitch):
        Span=sign[Ind:Ind+pitch]
        shrt=stretch(Span,coef)
        for I in shrt:
            res.append(I)
        K+=1
        if K>=distance:
            for I in shrt:
                res.append(I)
            K-=distance
    return res

rate, result = read("123.wav")
result = result.transpose()[0]
pitch=pitch_autocor(result,rate)
print(pitch)
res=pitch_change(0.6,1000,result)
z=max(res)
res/=z
res*=2**15
b = firwin(121, 3000/rate, pass_zero=True)
Z = sgn.lfilter(b, 1, res)
print(len(result),len(res))
write("result.wav",rate,np.array(res).astype(np.int16))

