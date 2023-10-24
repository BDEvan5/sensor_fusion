import numpy as np 
import matplotlib.pyplot as plt
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.MultiSensorRover import MultiSensorRobot
from sensor_fusion.estimators.NonLinearSensorFusion import NonLinearSensorFusion
from sensor_fusion.robots.vehicle_utils import *


simulation_time = 6 # seconds
f_control = 10 # seconds
f_mag = 5 # seconds
f_gps = 1 # seconds

def simulate_car_multiple(car, inputs, estimator_list):
    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        for estimator in estimator_list: 
            estimator.control_update(inputs[t])

        measurement = car.measure()
        for estimator in estimator_list: 
            estimator.measurement_update(measurement)
    
def test_full_fusion():
    np.random.seed(10)
    Q = np.diag([0.2**2, 0.2**2, 0.1**2])
    init_state = np.array([0,0,-np.pi/4])
    gps = LinearSensor("GPS", np.array([[1, 0, 0], [0, 1, 0]]), np.diag([0.1**2, 0.1**2]), f_gps)
    mag = LinearSensor("Mag", np.array([[0, 0, 1]]), np.diag([0.5]), f_mag)
    car = MultiSensorRobot(init_state, Q, 1/f_control, [gps, mag])
    init_belief = Gaussian(init_state, np.diag([1**2, 1**2, 0.5**2]))

    ekf_no_sensor = NonLinearSensorFusion(init_belief, car.f, car.F_k, Q, "No Sensor")
    ekf_mag_only = NonLinearSensorFusion(init_belief, car.f, car.F_k, Q, "Mag only", [mag])
    ekf_gps_only = NonLinearSensorFusion(init_belief, car.f, car.F_k, Q, "GPS only", [gps])
    ekf_full_fusion = NonLinearSensorFusion(init_belief, car.f, car.F_k, Q, "Full fusion", [gps, mag])
    estimator_list = [ekf_no_sensor, ekf_mag_only, ekf_gps_only, ekf_full_fusion]

    controls = np.sin(np.linspace(0, simulation_time, simulation_time*f_control) * 0.35)  *0.04

    simulate_car_multiple(car, controls, estimator_list)

    car_states = np.array(car.true_states)
    for estimator in estimator_list:
        print(f"Mean Error ({estimator.name}): {np.mean(np.abs(estimator.get_estimated_states() - car_states)):.3f}")

    plot_2D_comparison(estimator_list, car, "comparison")



def plot_2D_comparison(estimator_list, car, label):
    plt.figure(figsize=(12, 8))
    a1 = plt.subplot(2, 2, 1)
    a2 = plt.subplot(2, 2, 3)
    a3 = plt.subplot(2, 2, 2)
    a4 = plt.subplot(2, 2, 4)

    for f, estimator in enumerate(estimator_list):
        states = estimator.get_estimated_states()
        covariances = estimator.get_estimated_covariances()

        a1.plot(states[:, 0], states[:, 1], label=estimator.name)
        a2.plot(states[:, 2])

        pos_cov = (covariances[:, 0, 0]**2 + covariances[:, 1, 1]**2)**0.5
        a3.plot(pos_cov**0.5, label=estimator.name)
        a4.plot(covariances[:, 2, 2]**0.5)

    a1.set_aspect('equal')
    for a in [a1, a2, a3, a4]: a.grid(True)
    a1.set_ylabel("Position")
    a2.set_ylabel("Heading")
    a3.set_ylabel("Position Covariance")
    a4.set_ylabel("Heading Covariance")
    true_states = np.array(car.true_states)
    a1.plot(true_states[:, 0], true_states[:, 1], label="True Position", color='k')
    a2.plot(true_states[:, 2], label="True heading", color='k')
    a1.legend()
    a3.legend()

    plt.savefig(f"media/2D_car_{label}.svg")


test_full_fusion()

