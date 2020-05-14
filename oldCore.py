import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
osoba1 = loadmat('osoba_1.mat')['osoba_1']
osoba2 = loadmat('osoba_2.mat')['osoba_4']
# noinspection PyUnresolvedReferences
f, t, Zxx = signal.stft(osoba2[0, 1, 0, :], 1, signal.kaiser(256, 5), 256, 10, boundary=None)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.show()