import numpy as np
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.RoverRobot import RoverRobot
from sensor_fusion.utils.utils import *

from sensor_fusion.filters.ParticleFilter import ParticleFilter
from sensor_fusion.filters.UnscentedKalmanFilter import UnscentedKalmanFilter
from sensor_fusion.filters.ExtendedKalmanFilter import ExtendedKalmanFilter


Q = np.diag([0.2**2, 0.2**2, 0.05**2])
R = np.diag([1**2, 0.04**2])
f_s = 2
T = 25
init_state = np.array([0,0,-np.pi/4])
init_belief = Gaussian(init_state, np.diag([2**2,2**2,0.1**2]))

def simulate_rover_robot(robot, filter):
    controls = np.sin(np.arange(0, T*f_s+ 1, 1/f_s) * 0.2)  *0.08
    for k in range (1,T*f_s+1):
        control = controls[k]
        robot.move(control)
        filter.control_update(control)
        measurement = robot.measure()
        if measurement is not None:
            filter.measurement_update(measurement)


def test_ekf():
    ekf_robot = RoverRobot(init_state, Q, R, 1/f_s)
    ekf = ExtendedKalmanFilter(init_belief, ekf_robot.f, ekf_robot.h, ekf_robot.F_k, ekf_robot.H_k, Q, R)

    simulate_rover_robot(ekf_robot, ekf)

    plot_estimation_belief(ekf_robot, ekf)


def test_ukf():
    ukf_robot = RoverRobot(init_state, Q, R, 1/f_s)
    ukf = UnscentedKalmanFilter(init_belief, ukf_robot.f, ukf_robot.h, Q, R)

    simulate_rover_robot(ukf_robot, ukf)

    plot_estimation_belief(ukf_robot, ukf)

def test_pf():
    NP = 200
    pf_robot = RoverRobot(init_state, Q, R, 1/f_s)
    pf = ParticleFilter(init_belief, pf_robot.f, pf_robot.h, Q, R, NP)

    simulate_rover_robot(pf_robot, pf)

    plot_estimation(pf_robot, pf)


test_ekf()
# test_ukf()
# test_pf()
