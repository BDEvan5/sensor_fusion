import numpy as np

# Gaussian class
class Gaussian:
    # Initialisation
    def __init__(self, mean, cov):
        # insert code here
        self.mean = mean
        self.cov = cov
        self.N = mean.size

    # 1. Return the mean vector
    def get_mean(self):
        return self.mean
    
    # 2. Return the covariance matrix
    def get_covariance(self):
        return self.cov
    
    # 3. Draw P random samples in shape (P,N)
    def draw_samples(self, P):
        return np.random.multivariate_normal(self.mean, self.cov, P)
    
    # 4. Draw the 2N + 1 sigma points
    def draw_sigma_points(self, gamma):
        cov_pos = np.abs(self.cov)
        L = np.linalg.cholesky(cov_pos)
        sigmaPoints = np.concatenate((np.zeros((self.N,1)),gamma*L,-gamma*L), axis=1)
        return sigmaPoints.T + self.mean
    
    # 5. Return a new Gaussian marginalised to only the first N_ dimensions
    def marginalise(self, N_):
        return Gaussian(self.mean[:N_], self.cov[:N_,:N_])
    
    # 6. Return the parameters for plotting the k'th confidence ellipse
    def get_confidence_ellipse(self, k):
        U, s, Vh = np.linalg.svd(self.cov)
        if U[0][0]==0:
            angle = np.arctan(U[1][0]/0.001)
        else:
            angle = np.arctan(U[1][0]/U[0][0])
        return self.mean, k*np.sqrt(s[0]), k*np.sqrt(s[1]), angle