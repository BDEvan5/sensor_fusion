import numpy as np


def simulate_car(car, inputs, estimator, simulation_time, f_control):
    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        estimator.control_update(inputs[t])

        measurement = car.measure()
        estimator.measurement_update(measurement)

def simulate_car_multiple(car, inputs, filter_list, simulation_time, f_control):
    for t in range(simulation_time * f_control):
        car.move(inputs[t])
        for estimator in filter_list: 
            estimator.control_update(inputs[t])

        measurement = car.measure()
        for estimator in filter_list: 
            estimator.measurement_update(measurement)
    

class LinearSensor:
    def __init__(self, name, C, R, frequency) -> None:
        self.name = name
        self.C = C
        self.R = R
        self.dt = 1/frequency

    def measure(self, state, t):
        if (t+ 0.001 ) %self.dt< 1/20: 
            measurement =  self.C.dot(state) + np.random.multivariate_normal(np.zeros(self.R.shape[0]), self.R)
        else: measurement = None
        return measurement
    
