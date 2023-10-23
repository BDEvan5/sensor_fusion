import numpy as np


class StraightLineCar:
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