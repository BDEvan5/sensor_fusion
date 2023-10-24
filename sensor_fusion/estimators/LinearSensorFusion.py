import numpy as np 
from sensor_fusion.utils.Gaussian import Gaussian


class LinearSensorFusion:
    def __init__(self, init_belief, A, B, Q, name, sensor_list=[]):
        self.name = name
        self.beliefs = [init_belief]
        self.A = A
        self.B = B
        self.Q = Q

        self.sensors = sensor_list
        self.sensor_index = {sensor_list[i].name: i for i in range(len(sensor_list))}

    def control_update(self, u):
        mean = self.beliefs[-1].get_mean()
        cov = self.beliefs[-1].get_covariance()
        mean_ = self.A.dot(mean) + self.B.dot(u)
        cov_ = self.A.dot(cov).dot(self.A.T) + self.Q
        prior_belief = Gaussian(mean_, cov_)
        self.beliefs.append(prior_belief)

    def measurement_update(self, y):
        for m in y.keys():
            if y[m] is None: continue 
            if not m in self.sensor_index.keys(): continue
            C = self.sensors[self.sensor_index[m]].C
            R = self.sensors[self.sensor_index[m]].R

            mean = self.beliefs[-1].get_mean()
            cov = self.beliefs[-1].get_covariance()
            L = cov.dot(C.T).dot(np.linalg.inv(C.dot(cov).dot(C.T)+R))
            mean_ = mean + L.dot(y[m]-C.dot(mean))
            cov_ = (np.eye(mean.size)-L.dot(C)).dot(cov)
            self.beliefs[-1] = Gaussian(mean_, cov_)

    def get_estimated_states(self):
        states = [belief.get_mean() for belief in self.beliefs]
        return np.array(states)[:, :, 0]
    
    def get_estimated_covariances(self):
        covariances = [belief.get_covariance() for belief in self.beliefs]
        return np.array(covariances)
    
