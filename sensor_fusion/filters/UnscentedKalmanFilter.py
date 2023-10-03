from sensor_fusion.utils.Gaussian import Gaussian
import numpy as np


class UnscentedKalmanFilter:
    def __init__(self, init_belief, f, h, Q, R):
        self.beliefs = [init_belief]
        self.prior = None
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R

    def control_update(self, u):
        init_points = self.beliefs[-1].draw_sigma_points(1)
        final_points = self.f(init_points, u)
        mean, cov, cross = self.get_moments_from_sigma_points(init_points, np.array(final_points))
        self.prior = Gaussian(mean, cov + self.Q)

    def measurement_update(self, y):
        sigma_states = self.prior.draw_sigma_points(1)
        sigma_measurements = self.h(sigma_states)
        mean, cov, cross = self.get_moments_from_sigma_points(sigma_states, sigma_measurements)
        L = cross.dot(np.linalg.inv(cov+self.R))
        mean_ = self.prior.get_mean() + L.dot(y-mean)
        cov_ = self.prior.get_covariance() - L.dot(cov+self.R).dot(L.T)
        estimated_belief = Gaussian(mean_, cov_)
        self.beliefs.append(estimated_belief)

    def get_estimated_states(self):
        states = [belief.get_mean() for belief in self.beliefs]
        return np.array(states)
    
    def get_moments_from_sigma_points(self, init_points, final_points):
        n = init_points.shape[1]
        m = final_points.shape[1]
        mean = np.zeros(m)
        cov = np.zeros((m,m))
        cross = np.zeros((n,m))
        for i in range (2*n+1):
            if i == 0:
                w_m = 1 - n
            else:
                w_m = 1/2
            mean += w_m * final_points[i]
        for i in range (2*n+1):
            if i == 0:
                w_c = 4 - n - 1/n
            else:
                w_c = 1/2
            diff = final_points[i] - mean
            cov += w_c * np.outer(diff, diff)
            cross += w_c * np.outer(init_points[i] - init_points[0], diff)
        return mean, cov, cross
    