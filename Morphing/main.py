import numpy as np
from scipy.io import wavfile as wfl
from scipy import signal as sgn
[Fr,Dat] = wfl.read('123.wav') # женский голос
Dat = Dat.transpose()[0]
Len = 5000 # Длина фрагмента
Coe = 0.65 # Во сколько раз увеличиться pitch
Ln = len(Dat)
Out = np.zeros(Ln)
def stretch(In,Coe):
    LnFrame = len(In)
    LnOut = int(LnFrame * Coe)
    Out = np.zeros(LnOut)
    # print(In)
    for I in range(LnOut):
        Indx = int(I/Coe)
        Out[I] = In[Indx]
    return Out
Beg = 0
End = Beg + Len
Out = np.zeros(Ln)
R = int(Len * Coe)
wnd = sgn.hann(R)
Shift = int((Ln - R)*Len/(Ln -Len))
Pos = 0
while End <= Ln:
    Frame = np.float_(Dat[Beg:End])
    Res = stretch(Frame,Coe)
    Out[Pos:Pos + R] += .5 * Res * wnd
    Pos += Shift
    Beg += Len
    End = Beg + Len
wfl.write('Out.wav',int(Fr*1),np.int16(Out))