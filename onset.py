import numpy as np

def onsetDetect(sample):
    K = 0.7
    # Długość próbki
    sampleLen = 1500
    winLen = 60
    winInd = winLen
    onset = None

    mn0 = np.mean(abs(sample[0:winLen]))

    while winInd < len(sample) - winLen and onset is None:
        mnI = mn = np.mean(abs(sample[winInd:winInd + winLen]))

        if mnI > K * mn0:
            onset = winInd
        winInd += int(winLen / 2)
    return onset


def onsetCut(sample):
    K = 0.7
    # Długość próbki
    sampleLen = 1500
    winLen = 60
    winInd = winLen
    onset = None

    mn0 = np.mean(abs(sample[0:winLen]))

    while winInd < len(sample)-winLen and onset is None:
        mnI = mn = np.mean(abs(sample[winInd:winInd+winLen]))

        if mnI > K*mn0:
            onset = winInd
        winInd += int(winLen/2)
    return sample[onset:onset+sampleLen]



