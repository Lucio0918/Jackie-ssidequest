# % Class that implements principal components analysis on input data matrix
# % ECE 5370: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

# initialize with input data d, where d is N x M array where N
#   is the number of samples and M is the number of features
# Example usage:
# > d = np.random.rand(5,2) # 5 samples of a 2 feature process
# > p = pca(d)
# > d_pca = p.project(d) # returns coordinates of samples in the pca linear vector space

import numpy as np

class pca:
    def __init__(self, d):
        # compute and reshape mean feature vector
        self.mn = np.mean(d,0)[np.newaxis,:]
        # de-mean the features
        d_m = d - self.mn
        # compute covariance matrix
        covmat = d_m.T @ d_m/(len(d)-1)
        # compute eigenvalues and eigenvectors
        evals,evects = np.linalg.eig(covmat)
        # sort evals and evects according to largest magnitude eigenvalues
        i = np.argsort(-np.abs(evals))
        self.evals = evals[i]
        self.evects = evects[:,i]

    def project(self,d):
        # de-mean the features and project the residuals onto the eigenvectors, resulting in coordinates
        # in the PCA feature space
        d_m = d - self.mn
        f = d_m @ self.evects
        return f

  #  def num_effective_dims(self,percvar):



