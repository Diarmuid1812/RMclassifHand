import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
osoba1 = loadmat('osoba_1.mat')['osoba_1']
osoba2 = loadmat('osoba_2.mat')['osoba_4']

sig = osoba1[0, 0, 2, :]
t = np.arange(0, 2, 0.001)
# plt.plot(t, sig)
# plt.show()
K = 0.6
mn = np.mean(abs(sig))
fir = signal.firwin(60, 0.3, window='hamming')
filtered = signal.lfilter(fir, 0.1, sig)
plt.plot(t, filtered)
plt.show()
for pt in range(len(sig)-1):
    if filtered[pt+1] > 0.6:  # *mn (?)
        onset = pt
        break
