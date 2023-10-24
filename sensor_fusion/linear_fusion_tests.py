import numpy as np 
import matplotlib.pyplot as plt
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.StraightLineCar import StraightLineCar
from sensor_fusion.estimators.LinearSensorFusion import LinearSensorFusion
from sensor_fusion.robots.vehicle_utils import LinearSensor, simulate_car, simulate_car_multiple

simulation_time = 10 # seconds
f_control = 20 # seconds
f_imu = 10 # seconds
f_gps = 0.5 # seconds




def test_full_fusion():
    np.random.seed(10)
    init_state = np.array([[0], [0]])
    gps = LinearSensor("GPS", np.array([[1, 0]]), np.diag([0.1**2]), f_gps)
    imu = LinearSensor("IMU", np.array([[0, 1]]), np.diag([0.5**2]), f_imu)
    car = StraightLineCar(init_state, 1/f_control, [imu, gps])
    init_belief = Gaussian(init_state, np.diag([1**2, 0.5**2]))

    lkf_no_sensor = LinearSensorFusion(init_belief, car.A, car.B, car.motion_q, "No sensor")
    lkf_imu_only = LinearSensorFusion(init_belief, car.A, car.B, car.motion_q, "IMU Only", [imu])
    lkf_gps_only = LinearSensorFusion(init_belief, car.A, car.B, car.motion_q, "GPS Only", [gps])
    lkf_full_fusion = LinearSensorFusion(init_belief, car.A, car.B, car.motion_q, "Full fusion", [imu, gps])

    inputs = np.sin(np.linspace(0, 10, simulation_time * f_control))  + 0.02

    estimator_list = [lkf_no_sensor, lkf_imu_only, lkf_gps_only, lkf_full_fusion]
    simulate_car_multiple(car, inputs, estimator_list, simulation_time, f_control)

    car_states = np.array(car.true_states)[:, :, 0]
    for estimator in estimator_list:
        print(f"Mean Error ({estimator.name}): {np.mean(np.abs(estimator.get_estimated_states() - car_states)):.3f}")

    plot_comparison(car, estimator_list, "comparison")


def plot_comparison(car, estimator_list, save_name):
    plt.figure(figsize=(12, 8))
    a1 = plt.subplot(2, 2, 1)
    a2 = plt.subplot(2, 2, 3)
    a3 = plt.subplot(2, 2, 2)
    a4 = plt.subplot(2, 2, 4)

    for f, estimator in enumerate(estimator_list):
        states = estimator.get_estimated_states()
        covariances = estimator.get_estimated_covariances()

        a1.plot(states[:, 0], label=estimator.name)
        a2.plot(states[:, 1])
        a3.plot(covariances[:, 0, 0]**0.5, label=estimator.name)
        a4.plot(covariances[:, 1, 1]**0.5)

    for a in (a1, a2, a3, a4): a.grid(True)
    a1.set_ylabel("Position")
    a2.set_ylabel("Speed")
    a3.set_ylabel("Position Covariance")
    a4.set_ylabel("Velocity Covariance")
    true_states = np.array(car.true_states)
    a1.plot(true_states[:, 0], label="True Position", color='k')
    a2.plot(true_states[:, 1], label="True Velocity")
    a1.legend()
    a3.legend()

    plt.savefig(f"media/1D_car_{save_name}.svg")


test_full_fusion()

