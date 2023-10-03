import numpy as np
from numba import njit
from sensor_fusion.robots.ScanSimulator import ScanSimulator2D
from matplotlib import pyplot as plt

SPEED = 3
VEHICLE_LENGTH = 0.5
RANGE = 100

MAP_NAME = "aut_wide"
NUM_BEAMS = 10


class AutonomousRacer:
    def __init__(self, init_state, T_s):
        self.noisy_states = [init_state]
        self.true_states = [init_state]
        self.meaurements = []

        self.Q = np.diag([0.2**2]) # action noise 
        self.R = np.diag([0.2**2]) # Observation noise for the LiDAR. 
        self.T_s = T_s

        self.scan_simulator = ScanSimulator2D(MAP_NAME, NUM_BEAMS, np.pi)
        self.controller = RacingController()
        self.measure() # to correct number of measurements

    def move(self):
        u = self.controller.get_control(self.true_states[-1])
        true_state = dynamics(self.true_states[-1][None, :], u, self.T_s)[0, :]
        self.true_states.append(true_state)

        noisy_u = u + np.random.normal(0, self.Q[0, 0])
        noisy_state = dynamics(self.noisy_states[-1][None, :], noisy_u, self.T_s)[0, :]
        self.noisy_states.append(noisy_state)

        return noisy_u

    def measure(self):
        pose = self.true_states[-1]

        scan = self.scan_simulator.scan(pose)
        measurements = scan + np.random.normal(0, self.R[0], NUM_BEAMS)
        self.meaurements.append(measurements)

        return measurements
        
    def get_states(self):
        return np.array(self.true_states), np.array(self.noisy_states)

    def get_measurements(self):
        np.array(self.meaurements)
    
    def f(self, state, u):
        T_s = self.T_s
        new_state = dynamics(state, u, T_s)

        return new_state
    
    def h(self, states):
        scans = np.zeros((states.shape[0], NUM_BEAMS))
        for i, state in enumerate(states): 
            scans[i] = self.scan_simulator.scan(state)

        return scans



@njit(cache=True)
def dynamics(state, phi, T_s):
    new_state = np.zeros((state.shape[0], 3))
    new_state[:, 0] = state[:, 0] + T_s*SPEED*np.cos(state[:, 2])
    new_state[:, 1] = state[:, 1] + T_s*SPEED*np.sin(state[:, 2])
    new_state[:, 2] = state[:, 2] + T_s*SPEED*phi/VEHICLE_LENGTH

    return new_state


class RacingController:
    def __init__(self) -> None:
        pts = np.loadtxt(f"maps/{MAP_NAME}_centerline.csv", delimiter=',')

        seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        n_pts = int(np.sum(seg_lengths) * 10) # 10 cm level resolution
        
        old_ss = np.insert(np.cumsum(seg_lengths), 0, 0)
        new_ss = np.linspace(0, old_ss[-1], n_pts)

        self.wpts = np.zeros((n_pts, 2))
        self.wpts[:, 0] = np.interp(new_ss, old_ss, pts[:, 0])
        self.wpts[:, 1] = np.interp(new_ss, old_ss, pts[:, 1])

        self.wpts = np.row_stack((self.wpts, self.wpts[:50])) 

        self.current_ind = 0

    def get_control(self, pose):
        dists = np.linalg.norm(self.wpts[self.current_ind:self.current_ind+50] - pose[:2], axis=1)
        self.current_ind += np.argmin(dists)
        wpt = self.wpts[self.current_ind + 20] # 1.2 m lookahead distance

        steering = get_actuation(pose[2], wpt, pose[0:2], 1.2, VEHICLE_LENGTH) 
        theta_dot =  np.tan(steering) #/ VEHICLE_LENGTH

        return theta_dot



@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    if np.abs(waypoint_y) < 1e-6:
        return 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle

    

