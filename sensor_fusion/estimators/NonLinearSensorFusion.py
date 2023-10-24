import numpy as np 
from sensor_fusion.utils.Gaussian import Gaussian


class NonLinearSensorFusion:
    def __init__(self, init_belief, f, F_k, Q, name, sensor_list=[]):
        self.name = name
        self.beliefs = [init_belief]
        self.prior = None
        self.f = f
        self.F_k = F_k
        self.Q = Q

        self.sensors = sensor_list
        self.sensor_index = {sensor_list[i].name: i for i in range(len(sensor_list))}

    def add_sensor(self, sensor):
        self.sensors.append(sensor)
        self.sensor_index[sensor.name] = len(self.sensors)-1

    def control_update(self, u):
        mean = self.beliefs[-1].get_mean()
        cov = self.beliefs[-1].get_covariance()
        mean_ = self.f(mean, u)
        F = self.F_k(mean, u)
        cov_ = F.dot(cov).dot(F.T) + self.Q
        prior = Gaussian(mean_, cov_)
        self.beliefs.append(prior)

    def measurement_update(self, y):
        for m in y.keys():
            if y[m] is None: continue
            if not m in self.sensor_index.keys(): continue
            idx  = self.sensor_index[m]
            C = self.sensors[idx].C
            R = self.sensors[idx].R
            measurement = y[m]

            mean = self.beliefs[-1].get_mean()
            cov = self.beliefs[-1].get_covariance()
            L = cov.dot(C.T).dot(np.linalg.inv(C.dot(cov).dot(C.T)+R))
            mean_ = mean + L.dot(measurement-C.dot(mean))
            cov_ = (np.eye(mean.size)-L.dot(C)).dot(cov)
            self.beliefs[-1] = Gaussian(mean_, cov_)

    def get_estimated_states(self):
        states = [belief.get_mean() for belief in self.beliefs]
        return np.array(states)
    
    def get_estimated_covariances(self):
        covariances = [belief.get_covariance() for belief in self.beliefs]
        return np.array(covariances)
    