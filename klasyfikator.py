from scipy import io
from scipy import signal
import numpy as np
import pandas
#import matplotlib.pyplot as plt
path = 'C:/Users/Pawel/Desktop/Reka projekt/'
file = 'osoba_1.mat'
matdata = io.loadmat(path + file)
matdata.keys()
data = matdata['osoba_1']
data.shape

##### Struktura danych w plik .mat : ############

# 11 klas ruchów,
# 200 powtórzen kazdego ruchu,
# 16 kanałów (8xEMG, 8xMMG),
# 2000 odczytów na pomiar.

#################################################

nwf = 5 # zadana ilosc okien czestotliwosciowych
nwt = 5 # zadana ilosc okien czasowych 

L = []

for k in range(11) :
    for p in range(200) :
        F = np.zeros(201) # macierz do ktorej beda zapisywane cechy stft  i klasa ruchu
        for m in range(8) : # petla po 8 kanalach EMG
            u = data[k,p,m,:]
            t,f,z = signal.stft( u, nperseg = 2*np.round(np.sqrt(len(u))) )
            y = np.abs(z)
            nf = len(f) # ilosc okien czestotliwosciowych przed usrednieniem
            nt = len(t) # ilosc okien czasowych przed usrednieniem
            df = int(nf/nwf) # zadana dlugosc okna czestotliwosciowego
            dt = int(nt/nwt) # zadana dlugosc okna czasowego
            A = np.zeros( (nwf, nwt) )
            for i in range(nwf) :
                for j in range(nwt) :
                        A[i,j] = np.mean( y[i*df:(i+1)*df, j*dt:(j+1)*dt] )
            #plt.pcolormesh(range(nwf), range(nwt), A)
            #plt.show()
            F[m*25:(m+1)*25] = np.ndarray.flatten(A)
        F[200] = k # klasa ruchu (wartosc w zakresie od 0 do 10)
        L.append(F)

df = pandas.DataFrame(L)

################################################

# trzeba teraz podzielic zbior na dwie czesci
# zbior uczacy i zbior testowy

# uzycie jakiej selekcji cech, chociazby PCA
# wydaje mi sie konieczne (chyba ze zamiast tego jeszcze bardziej rozciagniemy
# okna w stft np. 3x3 zamiast 5x5, ale to raczej daloby kiepski efekt
# a 9*8 to wciaz jest duzo).

##################################################
