# % function to overlay a density scatter plot on a boxplot
# % ECE 5370: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import matplotlib.pyplot as plt
import numpy as np
def myBoxplot(D):
    plt.boxplot(D)

    sf = .1*(1-np.exp(-np.shape(D)[1]/4))
    d_bin = np.zeros(np.shape(D)[0],dtype=int)
    for i in range(0,np.shape(D)[1]):
        h,x = np.histogram(D[:,i])
        h = h/np.amax(h)
        for j in range(0, np.shape(D)[0]):
            bin = np.nonzero(D[j,i] < x)
            if np.size(bin)>0:
                d_bin[j] = bin[0][0]-1
            else:
                d_bin[j] = np.size(x)-2
        dx =np.random.default_rng(0).normal(size=np.shape(D)[0]) * sf * h[d_bin]
        msk = np.abs(dx) > sf * 2
        dx[msk] = 2 * sf * dx[msk] / np.abs(dx[msk])
        plt.plot(dx + i + 1, D[:, i], '.', color=[.5,.5,.5], alpha=.25)

