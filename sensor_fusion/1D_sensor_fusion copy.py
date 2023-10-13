import numpy as np 
import matplotlib.pyplot as plt
from sensor_fusion.utils.Gaussian import Gaussian


class OneDimensionalCar:
    def __init__(self, init_state, control_dt, imu_dt, gps_dt) -> None:
        self.true_states = [init_state]
        self.dead_reckon_states = [init_state]

        self.dt = control_dt
        self.imu_dt = imu_dt
        self.gps_dt = gps_dt
        
        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([[0], [self.dt]])

        self.C_imu = np.array([[0, 1]])
        self.C_gps = np.array([[1, 0]])

        self.imu_q = 0.2 # very noise
        self.gps_q = 0.05 # very accurate
        self.motion_q = np.diag([0.2, 0.05])


    def move(self, acceleration):
        dead_state = self.A.dot(self.dead_reckon_states[-1]) + self.B.dot(acceleration)
        self.dead_reckon_states.append(dead_state)

        true_state = self.A.dot(self.true_states[-1]) + self.B.dot(acceleration)
        true_state += np.random.multivariate_normal(np.zeros(2), self.motion_q)[:, None]
        self.true_states.append(true_state)

    def imu_measure(self):
        return self.C_imu.dot(self.dead_reckon_states[-1]) + np.random.normal(0, self.imu_q)
    
    def gps_measure(self):
        return self.C_gps.dot(self.dead_reckon_states[-1]) + np.random.normal(0, self.gps_q)
        


class SensorFusionLKF:
    def __init__(self, init_belief, A, B, C_imu, C_gps, Q, R_imu, R_gps):
        self.beliefs = [init_belief]
        self.prior = None
        self.A = A
        self.B = B
        self.C_imu = C_imu
        self.C_gps = C_gps
        self.Q = Q
        self.R_imu = R_imu
        self.R_gps = R_gps

    def control_update(self, u):
        mean = self.beliefs[-1].get_mean()
        cov = self.beliefs[-1].get_covariance()
        mean_ = self.A.dot(mean) + self.B.dot(u)
        cov_ = self.A.dot(cov).dot(self.A.T) + self.Q
        prior_belief = Gaussian(mean_, cov_)
        self.beliefs.append(prior_belief)

    def measurement_update_imu(self, y):
        mean = self.beliefs[-1].get_mean()
        cov = self.beliefs[-1].get_covariance()
        L = cov.dot(self.C.T).dot(np.linalg.inv(self.C.dot(cov).dot(self.C.T)+self.R))
        mean_ = mean + L.dot(y-self.C.dot(mean))
        cov_ = (np.eye(mean.size)-L.dot(self.C)).dot(cov)
        self.beliefs[-1] = Gaussian(mean_, cov_)

    def measurement_update_gps(self, y):
        mean = self.beliefs[-1].get_mean()
        cov = self.beliefs[-1].get_covariance()
        L = cov.dot(self.C.T).dot(np.linalg.inv(self.C.dot(cov).dot(self.C.T)+self.R))
        mean_ = mean + L.dot(y-self.C.dot(mean))
        cov_ = (np.eye(mean.size)-L.dot(self.C)).dot(cov)
        self.beliefs[-1] = Gaussian(mean_, cov_)

    def get_estimated_states(self):
        states = [belief.get_mean() for belief in self.beliefs]
        return np.array(states)[:, :, 0]
    
    def get_estimated_covariances(self):
        covariances = [belief.get_covariance() for belief in self.beliefs]
        return np.array(covariances)
    

simulation_time = 20 # seconds
f_control = 20 # seconds
f_imu = 10 # seconds
f_gps = 1 # seconds

    
def test_no_measurement():
    init_state = np.array([[0], [0]])
    car = OneDimensionalCar(init_state, 1/f_control, 1/f_imu, 1/f_gps)
    init_belief = Gaussian(init_state, np.diag([1**2, 0.5**2]))
    sensor_fusion_lkf = SensorFusionLKF(init_belief, car.A, car.B, car.C_imu, car.C_gps, car.motion_q, car.imu_q, car.gps_q)

    inputs = np.sin(np.arange(0, 10, car.dt) ) + 0.02

    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        sensor_fusion_lkf.control_update(inputs[t])

    plot_beliefs(sensor_fusion_lkf, car, inputs, "no_measurement")
    
def test_only_imu():
    init_state = np.array([[0], [0]])
    car = OneDimensionalCar(init_state, 1/f_control, 1/f_imu, 1/f_gps)
    init_belief = Gaussian(init_state, np.diag([1**2, 0.5**2]))
    sensor_fusion_lkf = SensorFusionLKF(init_belief, car.A, car.B, car.C_imu, car.C_gps, car.motion_q, car.imu_q, car.gps_q)

    inputs = np.sin(np.arange(0, 10, car.dt) ) + 0.02

    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        sensor_fusion_lkf.control_update(inputs[t])

        if (t%(f_control/f_imu) == 0):
            imu_measurement = car.imu_measurement()
            sensor_fusion_lkf.measurement_update_imu(imu_measurement)


    plot_beliefs(sensor_fusion_lkf, car, inputs, "no_measurement")



def test_no_measurement():
    init_state = np.array([[0], [0]])
    car = OneDimensionalCar(init_state)
    init_belief = Gaussian(init_state, np.diag([1**2, 0.5**2]))
    sensor_fusion_lkf = SensorFusionLKF(init_belief, car.A, car.B, car.C_imu, car.C_gps, car.motion_q, car.imu_q, car.gps_q)

    inputs = np.sin(np.arange(0, 10, car.dt) ) + 0.02

    simulation_time = 20 # seconds
    f_control = 20 # seconds
    f_imu = 10 # seconds
    f_gps = 1 # seconds

    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        sensor_fusion_lkf.control_update(inputs[t])

    plot_beliefs(sensor_fusion_lkf, car, inputs, "no_measurement")




def plot_beliefs(filter, car, inputs, label):
    plt.figure(figsize=(12, 8))
    a1 = plt.subplot(3, 2, 1)
    a2 = plt.subplot(3, 2, 3)
    a3 = plt.subplot(3, 2, 5)
    a4 = plt.subplot(3, 2, 2)
    a5 = plt.subplot(3, 2, 4)

    states = filter.get_estimated_states()
    covariances = filter.get_estimated_covariances()
    print(f"States: {states[50:60, 0]}")
    true_states = np.array(car.true_states)
    dead_reckon_states = np.array(car.dead_reckon_states)
    print(f"states: {states.shape}")
    a1.plot(filter.get_estimated_states()[:, 0], label="Estimated Position")
    a1.plot(true_states[:, 0], label="True Position")
    a1.plot(dead_reckon_states[:, 0], label="Dead Position")
    a1.legend()
    a1.set_ylabel("Position")
    a1.grid(True)
    a1.set_ylim(-4, 25)

    a2.plot(filter.get_estimated_states()[:, 1], label="Estimated Velocity")
    a2.plot(true_states[:, 1], label="True Velocity")
    a2.set_ylabel("Speed")
    a2.grid(True)
    a2.set_ylim(-4, 4)

    a3.plot(inputs, label="Inputs")
    a3.set_ylabel("Acceleration Input")
    a3.grid(True)

    a4.plot(covariances[:, 0, 0]**0.5, label="Covariances")
    a4.grid(True)
    a4.set_ylabel("Position Covariance")
    a4.set_ylim(0, 30)

    a5.plot(covariances[:, 1, 1]**0.5, label="Covariances")
    a5.grid(True)
    a5.set_ylabel("Velocity Covariance")
    a5.set_ylim(0, 4)

    plt.savefig(f"media/1D_car_{label}.svg")



test_no_measurement()
