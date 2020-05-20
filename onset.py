import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
osoba1 = loadmat('osoba_1.mat')['osoba_1']
osoba2 = loadmat('osoba_2.mat')['osoba_4']

sig = osoba1[0, 0, 4, :]
t = np.arange(0, 2, 0.001)
def onsetCut(sample):
    K = 1.5
    # Długość próbki
    sampleLen = 1500
    winLen = 60 #int(np.round(np.sqrt(len(sig))))
    winInd = winLen
    onset = None

    mn0 = np.mean(abs(sig[0:winLen]))

    while winInd < len(sig)-winLen and onset is None:
        mnI = mn = np.mean(abs(sig[winInd:winInd+winLen]))

        if mnI > K*mn0:
            onset = winInd
        winInd += int(winLen/2)
    return sample[onset:onset+sampleLen]
    # plt.plot(t, sig)
    # plt.show()
'''
K = 0.6
mn = np.mean(abs(sig))
fir = signal.firwin(60, 0.3, window='hamming')
filtered = signal.lfilter(fir, 0.1, sig)
# plt.plot(t, filtered)
# plt.show()
for pt in range(len(sig)-1):
    if filtered[pt+1] > 0.6:  # *mn (?)
        onset = pt
        break
'''

