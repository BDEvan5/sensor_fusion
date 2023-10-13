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
        self.motion_q = np.diag([0.2, 0.05])

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
        print(f"Measure {t%self.dt}")
        if t%self.dt < 0.05: 
            measurement =  self.C.dot(state) + np.random.multivariate_normal(np.zeros(1), self.R)
        else: measurement = None
        return measurement

simulation_time = 10 # seconds
f_control = 20 # seconds
f_imu = 10 # seconds
f_gps = 0.5 # seconds

    
def test_no_measurement():
    init_state = np.array([[0], [0]])
    car = OneDimensionalCar(init_state, 1/f_control)
    init_belief = Gaussian(init_state, np.diag([1**2, 0.5**2]))
    sensor_fusion_lkf = SensorFusionLKF(init_belief, car.A, car.B, car.motion_q)

    inputs = np.sin(np.linspace(0, 10, simulation_time * f_control))  + 0.02

    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        sensor_fusion_lkf.control_update(inputs[t])
        measurement = car.measure()
        sensor_fusion_lkf.measurement_update(measurement)

    plot_beliefs(sensor_fusion_lkf, car, inputs, "no_measurement")
    
def test_only_gps():
    init_state = np.array([[0], [0]])
    gps = GPS(np.array([[1, 0]]), np.diag([0.1**2]), f_gps)
    car = OneDimensionalCar(init_state, 1/f_control, [gps])
    init_belief = Gaussian(init_state, np.diag([1**2, 0.5**2]))
    sensor_fusion_lkf = SensorFusionLKF(init_belief, car.A, car.B, car.motion_q)
    sensor_fusion_lkf.add_sensor(gps)

    inputs = np.sin(np.linspace(0, 10, simulation_time * f_control))  + 0.02

    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        sensor_fusion_lkf.control_update(inputs[t])
        measurement = car.measure()
        sensor_fusion_lkf.measurement_update(measurement)

    plot_beliefs(sensor_fusion_lkf, car, inputs, "gps_only")



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
    a2.plot(dead_reckon_states[:, 1], label="Dead Position")
    a2.set_ylabel("Speed")
    a2.grid(True)
    a2.set_ylim(-4, 4)

    a3.plot(inputs, label="Inputs")
    a3.set_ylabel("Acceleration Input")
    a3.grid(True)

    a4.plot(covariances[:, 0, 0]**0.5, label="Covariances")
    a4.grid(True)
    a4.set_ylabel("Position Covariance")
    # a4.set_ylim(0, 30)

    a5.plot(covariances[:, 1, 1]**0.5, label="Covariances")
    a5.grid(True)
    a5.set_ylabel("Velocity Covariance")
    a5.set_ylim(0, 4)

    plt.savefig(f"media/1D_car_{label}.svg")



# test_no_measurement()
test_only_gps()

