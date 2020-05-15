from scipy import io
from scipy import signal
import numpy as np
#import pandas
import matplotlib.pyplot as plt
path = 'C:/Users/Pawel/Desktop/Reka projekt/'
file1 = 'osoba_1.mat'
file2 = 'osoba_2.mat'
#matdata1 = io.loadmat(path + file)
#matdata1.keys()
#osoba1 = matdata['osoba_1']
osoba1 = io.loadmat(path + file1)['osoba_1']
osoba2 = io.loadmat(path + file2)['osoba_4']
#osoba1.shape

##### Struktura danych w pliku .mat : ###########

# 11 klas ruchów,
# 200 powtórzen kazdego ruchu,
# 16 kanałów (8xEMG, 8xMMG),
# 2000 odczytów na pomiar.

################## FUNKCJE ######################

def averaged_stft_matrix( data, k, p, m, nwf, nwt, draw_stft=0, draw_signal=0 ) :
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
    if(draw_stft) :
        plt.pcolormesh(range(nwf), range(nwt), A)
        plt.show()
    if(draw_signal) :
        plt.plot(u)
    return A

def stft_features( data, k, p, nwf, nwt ) :
    F = np.zeros(200)
    for m in range(8) : # petla po 8 kanalach EMG
        F[m*25:(m+1)*25] = np.ndarray.flatten( averaged_stft_matrix(data,k,p,m,nwf,nwt) )
    return F

################## MAIN #########################
    
# Na poczatek przyjmuje, ze osoba1 tworzy zbior uczacy a osoba2 zbior testowy

nwf = 5 # zadana ilosc okien czestotliwosciowych
nwt = 5 # zadana ilosc okien czasowych

x_learn = []
y_learn = []
for k in range(11) :
    for p in range(200) :
        x_learn.append( stft_features(osoba1,k,p,nwf,nwt) ) # cechy stft
        y_learn.append( k ) # klasa ruchu (wartosc w zakresie od 0 do 10)

x_test = []
y_test = []
for k in range(11) :
    for p in range(200) :
        x_test.append( stft_features(osoba2,k,p,nwf,nwt) ) # cechy stft
        y_test.append( k ) # klasa ruchu (wartosc w zakresie od 0 do 10)

################################################

# Teraz trzeba wykonac selekcje cech metoda PCA oraz klasyfikacje metoda kNN

#from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

##################################################

# Najpierw klasyfikacja bez PCA :

kNN = KNeighborsClassifier() # domyslna liczba sasiadow = 5
kNN.fit( x_learn, y_learn )

# test na pojedynczym, pierwszym z brzegu sygnale :
kNN.predict( [ x_test[0] ] )
y_test[0]

# ocena klasyfikatora prez wyznaczenie sumarycznego
# bledu dopoasowania na zbiorze testowym :
y_test_pred = kNN.predict( x_test )
sum( [ 1 for i in range(len(y_test)) if y_test_pred[i] == y_test[i] ] )

##################################################
