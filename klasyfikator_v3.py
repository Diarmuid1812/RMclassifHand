import matplotlib
import notebook as notebook
from scipy import io
from scipy import signal
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
#import os
#os.chdir('C:/Users/Pawel/Desktop/Reka_projekt/')
from onset import onsetCut

path = './'
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
    u_Raw = data[k,p,m,:]
    u = onsetCut(u_Raw)

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
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')

        px = np.array([[i] * 5 for i in range(5)]).ravel()  # x coordinates of each bar
        py = np.array([i for i in range(5)] * 5)  # y coordinates of each bar
        z = np.zeros(5 * 5)  # z coordinates of each bar
        dx = np.ones(5 * 5)  # length along x-axis of each bar
        dy = np.ones(5 * 5)  # length along y-axis of each bar
        dz = A.ravel()  # length along z-axis of each bar (height)
        offset = dz + np.abs(dz.min())
        fracs = offset.astype(float) / offset.max()
        norm = col.Normalize(fracs.min(), fracs.max())
        cmap = cm.get_cmap('viridis')
        max_height = np.max(dz)  # get range of colorbars so we can normalize
        min_height = np.min(dz)
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k - min_height) / max_height) for k in dz]

        ha.bar3d(px, py, z, dx, dy, dz, color=rgba)
        plt.show()

    if(draw_signal) :
        plt.plot(u)
    return A

def stft_features( data, k, p, nwf, nwt ) :
    F = np.zeros(200)
    for m in range(8) : # petla po 8 kanalach EMG
        F[m*25:(m+1)*25] = np.ndarray.flatten( averaged_stft_matrix(data,k,p,m,nwf,nwt) )
    return F

################## Ekstrakcja cech ##############################

nwf = 5 # zadana ilosc okien czestotliwosciowych
nwt = 5 # zadana ilosc okien czasowych

x = [] # cechy stft
y = [] # klasa ruchu (wartosc w zakresie od 0 do 10)

for k in range(11) :
    for p in range(200) :
        x.append( stft_features(osoba1,k,p,nwf,nwt) ) 
        y.append( k ) 
for k in range(11) :
    for p in range(200) :
        x.append( stft_features(osoba2,k,p,nwf,nwt) )
        y.append( k )

# Losowy podzial na zbiory :
        
test_percentage = 10 # procentowy udzial zbioru testowego [%]

test_quantity = int(len(x)*test_percentage/100)
test_indices = random.sample(range(len(x)),test_quantity)

# Zbior uczacy :

x_learn =  [ x[i] for i in range(len(x)) if i not in test_indices ]
y_learn =  [ y[i] for i in range(len(x)) if i not in test_indices ]
        

# Zbior testowy :

x_test = [ x[i] for i in test_indices ]
y_test = [ y[i] for i in test_indices ]


#################################################################
#### Zaladowanie pakietow do selekcji cech i klasyfikacji #######

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#################################################################

# Najpierw kNN bez PCA ale z onsetem :

kNN_Raw = KNeighborsClassifier() # domyslna liczba sasiadow = 5
kNN_Raw.fit( x_learn, y_learn )

# ocena klasyfikatora na zbiorze uczacym
y_learnPred_Raw = kNN_Raw.predict( x_learn )
Valid_learn_Raw = sum( [ 1 for i in range(len(y_learn)) if y_learnPred_Raw[i] == y_learn[i] ] )
print(Valid_learn_Raw/len(y_learn))

# ocena klasyfikatora na zbiorze testowym :
y_testPred_Raw = kNN_Raw.predict( x_test )
Valid_test_Raw = sum( [ 1 for i in range(len(y_test)) if y_testPred_Raw[i] == y_test[i] ] )
print(Valid_test_Raw/len(y_test))

##################################################

# teraz kNN z PCA i onsetem :

# PCA
# variance explained to >99% (parametr z pracy dypl p. Boczara)
# przegląd zupełny cech, konieczny do operowania na wariancji
pca = PCA(0.99, svd_solver='full')

# zbiór uczacy dla kNN w zrzutowanej przestrzeni
x_learnArr = np.vstack(x_learn)
x_learnNorm = StandardScaler().fit_transform(x_learnArr)

pca.fit(x_learnNorm)

x_learnPCA = pca.transform(x_learnNorm)
x_learnPCA_prepared = np.vsplit(x_learnPCA, x_learnPCA.shape[0])
for g in range(x_learnPCA.shape[0]):
    x_learnPCA_prepared[g] = x_learnPCA_prepared[g].reshape(-1)

# Zapisanie liczby cech dla rzutowań nowych próbek
#components = pca.components_[0]
# skonstuowanie nowego PCA -- ze stałą liczbą cech

# kNN
kNN = KNeighborsClassifier()
kNN.fit(x_learnPCA_prepared, y_learn)

import pickle
knnPickle = open('./modelPZEZMS.pkl', 'wb')

pickle.dump(kNN, knnPickle)

# ocena klasyfikatora na zbiorze uczacym
y_learn_pred = kNN.predict( x_learnPCA_prepared )
Valid_learn = sum( [ 1 for i in range(len(y_learn)) if y_learn_pred[i] == y_learn[i] ] )
print(Valid_learn/len(y_learn))

# zbiór testowy dla kNN w zrzutowanej przestrzeni
x_testArr = np.vstack(x_test)
x_testNorm = StandardScaler().fit_transform(x_testArr)

x_testPCA = pca.transform(x_testNorm)
x_testPCA_prepared = np.vsplit(x_testPCA, x_testPCA.shape[0])
for g in range(x_testPCA.shape[0]):
    x_testPCA_prepared[g] = x_testPCA_prepared[g].reshape(-1)

y_test_pred = kNN.predict(x_testPCA_prepared)

# ocena klasyfikatora na zbiorze testowym :
Valid_test = sum( [ 1 for i in range(len(y_test)) if y_test_pred[i] == y_test[i] ] )
print(Valid_test/len(y_test))

# END

##################################################

# do wyprobowania wieloklasowy klasyfikator SVM :

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
#OneVsRestClassifier(SVC().fit(X,y))

svm_Raw = OneVsRestClassifier(SVC()).fit(x_learn, y_learn)
svm = OneVsRestClassifier(SVC()).fit(x_learnPCA_prepared, y_learn)

# ocena klasyfikatora na zbiorze uczacym bez PCA :
y_learnPred_svm_Raw = svm_Raw.predict( x_learn )
Valid_learn_svm_Raw = sum( [ 1 for i in range(len(y_learn)) if y_learnPred_svm_Raw[i] == y_learn[i] ] )
print(Valid_learn_svm_Raw/len(y_learn))

# ocena klasyfikatora na zbiorze testowym bez PCA :
y_testPred_svm_Raw = svm_Raw.predict( x_test )
Valid_test_svm_Raw = sum( [ 1 for i in range(len(y_test)) if y_testPred_svm_Raw[i] == y_test[i] ] )
print(Valid_test_svm_Raw/len(y_test))

# ocena klasyfikatora na zbiorze uczacym
y_learnPred_svm = svm.predict( x_learnPCA_prepared )
Valid_learn_svm = sum( [ 1 for i in range(len(y_learn)) if y_learnPred_svm[i] == y_learn[i] ] )
print(Valid_learn_svm/len(y_learn))

# ocena klasyfikatora na zbiorze testowym :
y_testPred_svm = svm.predict( x_testPCA_prepared )
Valid_test_svm = sum( [ 1 for i in range(len(y_test)) if y_testPred_svm[i] == y_test[i] ] )
print(Valid_test_svm/len(y_test))

#################################################
