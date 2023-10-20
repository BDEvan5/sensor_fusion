import numpy as np 
import matplotlib.pyplot as plt
from sensor_fusion.utils.Gaussian import Gaussian


class OneDimensionalCar:
    def __init__(self, init_state, control_dt, sensors=[]) -> None:
        self.true_states = [init_state]
        self.dead_reckon_states = [init_state]

        self.time = 0
        self.dt = control_dt
        self.sensors = sensors
        
        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([[0], [self.dt]])
        self.motion_q = np.diag([0.5, 0.05])

    def move(self, acceleration):
        dead_state = self.A.dot(self.dead_reckon_states[-1]) + self.B.dot(acceleration)
        self.dead_reckon_states.append(dead_state)

        true_state = self.A.dot(self.true_states[-1]) + self.B.dot(acceleration)
        true_state += np.random.multivariate_normal(np.zeros(2), self.motion_q)[:, None]
        self.true_states.append(true_state)

        self.time += self.dt

    def measure(self):
        measurements = {}
        for sensor in self.sensors:
            m = sensor.measure(self.true_states[-1], self.time)
            measurements[sensor.name] = m

        return measurements


class SensorFusionLKF:
    def __init__(self, init_belief, A, B, Q):
        self.beliefs = [init_belief]
        self.prior = None
        self.A = A
        self.B = B
        self.Q = Q

        self.sensors = []
        self.sensor_index = {}

    def add_sensor(self, sensor):
        self.sensors.append(sensor)
        self.sensor_index[sensor.name] = len(self.sensors)-1

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
        return np.array(states)[:, :, 0]
    
    def get_estimated_covariances(self):
        covariances = [belief.get_covariance() for belief in self.beliefs]
        return np.array(covariances)
    

class GPS:
    def __init__(self, C, R, frequency) -> None:
        self.name = "GPS"
        self.C = C
        self.R = R
        self.dt = 1/frequency

    def measure(self, state, t):
        if t%self.dt < 0.05: 
            measurement =  self.C.dot(state) + np.random.multivariate_normal(np.zeros(1), self.R)
        else: measurement = None
        return measurement


class IMU:
    def __init__(self, C, R, frequency) -> None:
        self.name = "IMU"
        self.C = C
        self.R = R
        self.dt = 1/frequency

    def measure(self, state, t):
        if t%self.dt < 0.05: 
            measurement =  self.C.dot(state) + np.random.multivariate_normal(np.zeros(1), self.R)
        else: measurement = None
        return measurement


simulation_time = 10 # seconds
f_control = 20 # seconds
f_imu = 10 # seconds
f_gps = 0.5 # seconds


def test_full_fusion():
    np.random.seed(10)
    init_state = np.array([[0], [0]])
    gps = GPS(np.array([[1, 0]]), np.diag([0.1**2]), f_gps)
    imu = IMU(np.array([[0, 1]]), np.diag([0.5**2]), f_imu)
    car = OneDimensionalCar(init_state, 1/f_control, [imu, gps])
    init_belief = Gaussian(init_state, np.diag([1**2, 0.5**2]))

    lkf_no_sensor = SensorFusionLKF(init_belief, car.A, car.B, car.motion_q)
    lkf_imu_only = SensorFusionLKF(init_belief, car.A, car.B, car.motion_q)
    lkf_imu_only.add_sensor(imu)
    lkf_gps_only = SensorFusionLKF(init_belief, car.A, car.B, car.motion_q)
    lkf_gps_only.add_sensor(gps)
    lkf_full_fusion = SensorFusionLKF(init_belief, car.A, car.B, car.motion_q)
    lkf_full_fusion.add_sensor(imu)
    lkf_full_fusion.add_sensor(gps)

    inputs = np.sin(np.linspace(0, 10, simulation_time * f_control))  + 0.02

    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        lkf_no_sensor.control_update(inputs[t])
        lkf_imu_only.control_update(inputs[t])
        lkf_gps_only.control_update(inputs[t])
        lkf_full_fusion.control_update(inputs[t])

        measurement = car.measure()
        lkf_no_sensor.measurement_update(measurement)
        lkf_imu_only.measurement_update(measurement)
        lkf_gps_only.measurement_update(measurement)
        lkf_full_fusion.measurement_update(measurement)

    car_states = np.array(car.true_states)[:, :, 0]
    print(f"Mean Error (no sensor): {np.mean(np.abs(lkf_no_sensor.get_estimated_states() - car_states)):.3f}")
    print(f"Mean Error (imu only): {np.mean(np.abs(lkf_imu_only.get_estimated_states() - car_states)):.3f}")
    print(f"Mean Error (gps only): {np.mean(np.abs(lkf_gps_only.get_estimated_states() - car_states)):.3f}")
    print(f"Mean Error (full fusion): {np.mean(np.abs(lkf_full_fusion.get_estimated_states() - car_states)):.3f}")

    plot_comparison(lkf_no_sensor, lkf_imu_only, lkf_gps_only, lkf_full_fusion, car, "comparison")



def plot_comparison(none, imu, gps, full, car, label):
    plt.figure(figsize=(12, 8))
    a1 = plt.subplot(2, 2, 1)
    a2 = plt.subplot(2, 2, 3)
    a3 = plt.subplot(2, 2, 2)
    a4 = plt.subplot(2, 2, 4)

    filter_list = [none, imu, gps, full]
    labels = ["No Sensor", "IMU Only", "GPS Only", "Full Fusion"]

    for f, filter in enumerate(filter_list):
        states = filter.get_estimated_states()
        covariances = filter.get_estimated_covariances()

        a1.plot(states[:, 0], label=labels[f])
        a2.plot(states[:, 1])
        a3.plot(covariances[:, 0, 0]**0.5, label=labels[f])
        a4.plot(covariances[:, 1, 1]**0.5)

    a1.set_ylim(-4, 25)
    a2.set_ylim(-4, 4)
    a4.set_ylim(0, 4)
    a1.grid(True)
    a2.grid(True)
    a3.grid(True)
    a4.grid(True)
    a1.set_ylabel("Position")
    a2.set_ylabel("Speed")
    a3.set_ylabel("Position Covariance")
    a4.set_ylabel("Velocity Covariance")
    true_states = np.array(car.true_states)
    a1.plot(true_states[:, 0], label="True Position", color='k')
    a2.plot(true_states[:, 1], label="True Velocity")
    a1.legend()
    a3.legend()

    plt.savefig(f"media/1D_car_{label}.svg")


test_full_fusion()

