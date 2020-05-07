from scipy import io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
#path = 'C:/Users/Pawel/Desktop/Reka projekt/'
path = './'
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
# przykladowe wyznaczenie cech stft dla pojedyneczego przebiegu sygnalu:
#################################################
u = data[0,0,0,:]
t,f,z = signal.stft( u, nperseg = 2*np.round(np.sqrt(len(u))), boundary=None)
y = np.abs(z)
nf = len(f) # ilosc okien czestotliwosciowych przed usrednieniem
nt = len(t) # ilosc okien czasowych przed usrednieniem
nwf = 5 # zadana ilosc okien czestotliwosciowych
nwt = 5 # zadana ilosc okien czasowych 
df = int(nf/nwf) # zadana dlugosc okna czestotliwosciowego
dt = int(nt/nwt) # zadana dlugosc okna czasowego
A = np.zeros( (nwf, nwt) )
for i in range(nwf) :
    for j in range(nwt) :
        A[i,j] = np.mean( y[i*df:(i+1)*df, j*dt:(j+1)*dt] )
plt.pcolormesh(range(nwf), range(nwt), A)
plt.show()
# przy liczeniu sredniej trzeba wziac modul liczby zespolonej
        
################################################
        
# jak dostaniemy A w postaci macierzy 5x5 to potem trzeba zagregowac
# po kanalach EMG, czyli dostaniemy 25*8 = 200 wartosci ktore traktujemy
# dalej jako cechy do algorytmu klasyfikacji

# potem jakos trzeba zebrac dane do tablicy gdzie bedziemy mieli
# 200 kolumn z cechami, ktore wzynaczylismy i 1 kolumne o nazwie
# klasa ruchu (z wartosciami od 1 do 11)
# wierszy powinno byc 200*11 = 2200
# jak to bedziemy mieli to wtedy pozostaje podzielic zbior na dwie czesci
# zbior uczacy i zbior testowy
        
# aha no i w sumie uzycie jakiej selekcji cech, chociazby PCA
# wydaje mi sie konieczne (chyba ze zamiast tego jeszcze bardziej rozciagniemy
# okna w stft np. 3x3 zamiast 5x5, ale to raczej daloby kiepski efekt
# a 9*8 to wciaz jest duzo).

##################################################
        