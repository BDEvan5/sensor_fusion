import numpy as np
from numba import njit

SPEED = 6
VEHICLE_LENGTH = 1
RANGE = 100


class MultiSensorRobot:
    def __init__(self, init_state, Q, T_s, sensors):
        self.deadreckon_states = [init_state]
        self.true_states = [init_state]
        self.Q = Q
        self.sensors = sensors
        self.dT = T_s
        self.time = 0
        self.measurements = []
        self.measure() # correct number.

    def move(self, u):
        deadreckon_state = dynamics(self.deadreckon_states[-1][None, :], u, self.dT)[0, :]
        self.deadreckon_states.append(deadreckon_state)

        true_state = dynamics(self.true_states[-1][None, :], u, self.dT)[0, :]
        true_state += np.random.multivariate_normal(np.zeros(3),self.Q)
        self.true_states.append(true_state)
        self.time += self.dT

    def get_states(self):
        return np.array(self.true_states), np.array(self.deadreckon_states)
    
    def f(self, state, u):
        if len(state.shape) == 1: # if state is 1D
            new_state = dynamics(state[None, :], u, self.dT)[0, :]
        else:
            new_state = dynamics(state, u, self.dT)
        return new_state
        
    def F_k(self, state, u):
        F = np.eye(3)
        F[0][2] = -self.dT*SPEED*np.sin(state[2])
        F[1][2] = self.dT*SPEED*np.cos(state[2])
        return F
    
    def measure(self):
        measurements = {}
        for sensor in self.sensors:
            m = sensor.measure(self.true_states[-1], self.time)
            measurements[sensor.name] = m

        return measurements


@njit(cache=True)
def dynamics(state, phi, T_s):
    new_state = np.zeros((state.shape[0], 3))
    new_state[:, 0] = state[:, 0] + T_s*SPEED*np.cos(state[:, 2])
    new_state[:, 1] = state[:, 1] + T_s*SPEED*np.sin(state[:, 2])
    new_state[:, 2] = state[:, 2] + T_s*SPEED*phi/VEHICLE_LENGTH

    return new_state
