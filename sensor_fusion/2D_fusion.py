import numpy as np 
import matplotlib.pyplot as plt
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.MultiSensorRover import MultiSensorRobot, GPS, IMU, Magnotometer



class SensorFusionEKF:
    def __init__(self, init_belief, f, F_k, Q):
        self.beliefs = [init_belief]
        self.prior = None
        self.f = f
        self.F_k = F_k
        self.Q = Q

        self.sensors = []
        self.sensor_index = {}

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
    

simulation_time = 20 # seconds
f_control = 20 # seconds
f_imu = 10 # seconds
f_gps = 0.5 # seconds


def test_full_fusion():
    Q = np.diag([0.2**2, 0.2**2, 0.05**2])
    # np.random.seed(10)
    init_state = np.array([0,0,-np.pi/4])
    gps = GPS(np.array([[1, 1, 0]]), np.diag([0.1**2]), f_gps)
    mag = Magnotometer(np.array([[0, 0, 1]]), np.diag([0.3**2]), f_imu)
    car = MultiSensorRobot(init_state, Q, 1/f_control, [gps, mag])
    init_belief = Gaussian(init_state, np.diag([1**2, 1**2, 0.5**2]))

    ekf_no_sensor = SensorFusionEKF(init_belief, car.f, car.F_k, Q)
    ekf_mag_only = SensorFusionEKF(init_belief, car.f, car.F_k, Q)
    ekf_mag_only.add_sensor(mag)
    ekf_gps_only = SensorFusionEKF(init_belief, car.f, car.F_k, Q)
    ekf_gps_only.add_sensor(gps)
    ekf_full_fusion = SensorFusionEKF(init_belief, car.f, car.F_k, Q)
    ekf_full_fusion.add_sensor(mag)
    ekf_full_fusion.add_sensor(gps)

    controls = np.sin(np.linspace(0, simulation_time, simulation_time*f_control) * 0.35)  *0.04

    for t in range(simulation_time * f_control):
        car.move(controls[t])
        ekf_no_sensor.control_update(controls[t])
        ekf_mag_only.control_update(controls[t])
        ekf_gps_only.control_update(controls[t])
        ekf_full_fusion.control_update(controls[t])

        measurement = car.measure()
        ekf_no_sensor.measurement_update(measurement)
        ekf_mag_only.measurement_update(measurement)
        ekf_gps_only.measurement_update(measurement)
        ekf_full_fusion.measurement_update(measurement)

    car_states = np.array(car.true_states)
    print(f"Mean Error (no sensor): {np.mean(np.abs(ekf_no_sensor.get_estimated_states() - car_states)):.3f}")
    print(f"Mean Error (mag only): {np.mean(np.abs(ekf_mag_only.get_estimated_states() - car_states)):.3f}")
    print(f"Mean Error (gps only): {np.mean(np.abs(ekf_gps_only.get_estimated_states() - car_states)):.3f}")
    print(f"Mean Error (full fusion): {np.mean(np.abs(ekf_full_fusion.get_estimated_states() - car_states)):.3f}")

    plot_2D_comparison([ekf_no_sensor, ekf_gps_only, ekf_mag_only, ekf_full_fusion], car, "comparison")



def plot_2D_comparison(filter_list, car, label):
    plt.figure(figsize=(12, 8))
    a1 = plt.subplot(2, 2, 1)
    a2 = plt.subplot(2, 2, 3)
    a3 = plt.subplot(2, 2, 2)
    a4 = plt.subplot(2, 2, 4)

    labels = ["No Sensor", "GPS Only", "Mag Only", "Full Fusion"]

    for f, filter in enumerate(filter_list):
        states = filter.get_estimated_states()
        covariances = filter.get_estimated_covariances()

        a1.plot(states[:, 0], states[:, 1], label=labels[f])
        a2.plot(states[:, 2])

        x_cov = covariances[:, 0, 0]
        y_cov = covariances[:, 1, 1]
        pos_cov = (x_cov**2 + y_cov**2)**0.5
        a3.plot(pos_cov**0.5, label=labels[f])
        a4.plot(covariances[:, 2, 2]**0.5)

    a1.set_aspect('equal')
    a1.grid(True)
    a2.grid(True)
    a3.grid(True)
    a4.grid(True)
    a1.set_ylabel("Position")
    a2.set_ylabel("Heading")
    a3.set_ylabel("Position Covariance")
    a4.set_ylabel("Heading Covariance")
    true_states = np.array(car.true_states)
    a1.plot(true_states[:, 0], true_states[:, 1], label="True Position", color='k')
    a2.plot(true_states[:, 2], label="True heading", color='k')
    a1.legend()
    a3.legend()

    plt.savefig(f"media/1D_car_{label}.svg")


test_full_fusion()

