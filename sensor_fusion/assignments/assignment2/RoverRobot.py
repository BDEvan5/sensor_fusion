import numpy as np
from numba import njit

SPEED = 3
VEHICLE_LENGTH = 1
RANGE = 100


class RoverRobot:
    def __init__(self, init_state, Q, R, T_s):
        self.deadreckon_states = [init_state]
        self.true_states = [init_state]
        self.Q = Q
        self.R = R
        self.T_s = T_s
        self.beacon = np.array([20, 15])
        # self.beacon = np.array([50, 50])
        self.measurements = []
        self.measure() # correct number.

    def move(self, u):
        T_s = self.T_s
        deadreckon_state = dynamics(self.deadreckon_states[-1][None, :], u, T_s)[0, :]
        self.deadreckon_states.append(deadreckon_state)

        true_state = dynamics(self.true_states[-1][None, :], u, T_s)[0, :]
        true_state += np.random.multivariate_normal(np.zeros(3),self.Q)
        self.true_states.append(true_state)

    def measure(self):
        x_B, y_B = self.beacon
        x, y, theta = self.true_states[-1]
        r = np.sqrt((x-x_B)**2+(y-y_B)**2)
        beta = np.arctan2(y-y_B,x-x_B)-theta
        y = np.array([r, beta]) 
        measurement = y + np.random.multivariate_normal(np.zeros(2),self.R)
        self.measurements.append(measurement)

        return measurement
        
    def get_states(self):
        return np.array(self.true_states), np.array(self.deadreckon_states)
    
    def f(self, state, u):
        if len(state.shape) == 1: # if state is 1D
            new_state = dynamics(state[None, :], u, self.T_s)[0, :]
        else:
            new_state = dynamics(state, u, self.T_s)
        return new_state
    
    def h(self, states):
        x_B, y_B = self.beacon
        if len(states.shape) == 1:
            states = states[None, :]
        x, y, theta = np.hsplit(states, 3)
        r = np.sqrt((x-x_B)**2+(y-y_B)**2)
        beta = np.arctan2(y-y_B,x-x_B)-theta
        y = np.concatenate([r, beta], axis=1)
        if y.shape[0] == 1:
            y = y[0]

        return y
    
    def F_k(self, state, u):
        T_s = self.T_s
        x, y, theta = state
        F = np.eye(3)
        F[0][2] = -T_s*SPEED*np.sin(theta)
        F[1][2] = T_s*SPEED*np.cos(theta)
        return F
    
    def H_k(self, state):
        x, y, theta = state
        delta_x = x - self.beacon[0]
        delta_y = y - self.beacon[1]
        r = np.sqrt(delta_x**2+delta_y**2)
        H = np.zeros((2,3))
        H[0][0] = delta_x/r
        H[0][1] = delta_y/r
        # H[1][0] = 1/(delta_x+delta_y**2/delta_x)
        H[1][0] = -delta_y/(delta_x**2+delta_y**2)
        # H[1][1] = -1/(delta_y+delta_x**2/delta_y)
        H[1][1] = delta_x/(delta_y**2+delta_x**2)
        H[1][2] = -1
        return H


@njit(cache=True)
def dynamics(state, phi, T_s):
    new_state = np.zeros((state.shape[0], 3))
    new_state[:, 0] = state[:, 0] + T_s*SPEED*np.cos(state[:, 2])
    new_state[:, 1] = state[:, 1] + T_s*SPEED*np.sin(state[:, 2])
    new_state[:, 2] = state[:, 2] + T_s*SPEED*phi/VEHICLE_LENGTH

    return new_state


