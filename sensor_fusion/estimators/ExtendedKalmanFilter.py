from sensor_fusion.utils.Gaussian import Gaussian
import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, init_belief, f, h, F_k, H_k, Q, R):
        self.name = "Extended Kalman Filter"
        self.beliefs = [init_belief]
        self.prior = None
        self.f = f
        self.h = h
        self.F_k = F_k
        self.H_k = H_k
        self.Q = Q
        self.R = R

    def control_update(self, u):
        mean = self.beliefs[-1].get_mean()
        cov = self.beliefs[-1].get_covariance()
        mean_ = self.f(mean, u)
        F = self.F_k(mean, u)
        cov_ = F.dot(cov).dot(F.T) + self.Q
        self.prior = Gaussian(mean_, cov_)

    def measurement_update(self, y):
        mean = self.prior.get_mean()
        cov = self.prior.get_covariance()
        H = self.H_k(mean)
        L = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov).dot(H.T)+self.R))
        mean_ = mean + L.dot(y-self.h(mean))
        cov_ = (np.eye(mean.size)-L.dot(H)).dot(cov)
        posterior = Gaussian(mean_, cov_)
        self.beliefs.append(posterior)
    
    def get_estimated_states(self):
        states = [belief.get_mean() for belief in self.beliefs]
        return np.array(states)
    
    