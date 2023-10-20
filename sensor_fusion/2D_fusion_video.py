import numpy as np 
import matplotlib.pyplot as plt
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.MultiSensorRover import MultiSensorRobot, GPS, IMU, Magnotometer
from matplotlib.animation import FuncAnimation


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
    

simulation_time = 7 # seconds
f_control = 10 # seconds
f_imu = 5 # seconds
f_gps = 1 # seconds


def test_full_fusion():
    np.random.seed(30)
    Q = np.diag([0.01, 0.01, 0.02])
    # Q = np.diag([0.2**2, 0.2**2, 0.05**2])
    # np.random.seed(10)
    init_state = np.array([22,22,-np.pi/6])
    gps = GPS(np.array([[1, 1, 0]]), np.diag([0.01**2]), f_gps)
    mag = Magnotometer(np.array([[0, 0, 1]]), np.diag([1]), f_imu)
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

    controls = np.sin(np.linspace(0, simulation_time, simulation_time*f_control) *1)  *0.1
    # controls = np.sin(np.linspace(0, simulation_time, simulation_time*f_control) *1)  *0.1
    controls = -0.1 * np.ones(simulation_time * f_control)

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

from matplotlib import patches as pat

from matplotlib.patches import Rectangle
def plot_2D_comparison(filter_list, car, label):
    fig = plt.figure(figsize=(10, 5))
    a1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    a2 = plt.subplot2grid((2, 2), (0, 1))
    a3 = plt.subplot2grid((2, 2), (1, 1))

    labels = ["No Sensor", "GPS Only", "Magnotometer Only", "GPS + Magnotometer"]
    state_list = [filter.get_estimated_states() for filter in filter_list]
    cov_list = [filter.get_estimated_covariances() for filter in filter_list]
    pos_cov_list = [(cov[:, 0, 0]**2 + cov[:, 1, 1]**2)**0.5 for cov in cov_list]
    # belief_list = [filter.beliefs for filter in filter_list]
    full_beliefs = filter_list[3].beliefs

    mean, major, minor, angle = full_beliefs[0].marginalise(2).get_confidence_ellipse(1)
    arc = pat.Arc((mean[0],mean[1]),major,minor,angle, ec="#fa8231", color="#fa8231", linewidth=5)
    a2.add_patch(arc)
    head_cov_list = [cov[:, 2, 2]**0.5 for cov in cov_list]
    true_states = np.array(car.true_states)
    print(f"True: {true_states[:5]},")

    pos_lines = []
    pos_cov_lines = []
    head_cov_lines = []
    rect = Rectangle((25, 25), 3, 1, color='lightgrey')
    # rect = Rectangle((25, 25), 3, 1, rotation_point='center', color='lightgrey')
    # a1.add_patch(rect)
    colors = ['#f7b731', '#45aaf2', '#20bf6b', '#eb3b5a']
    # colors = ['#a55eea', '#45aaf2', '#20bf6b', '#eb3b5a']
    for i in range(4):
        color = colors[i]
        pos_line = a1.plot(state_list[i][0, 0], state_list[i][0, 1], label=labels[i], color=color, linewidth=2)
        pos_lines.append(pos_line)
        # pos_cov_line = a2.plot(pos_cov_list[i], color=color)
        # pos_cov_lines.append(pos_cov_line)
        head_cov_line = a3.plot(cov_list[i][0, 2, 2]**0.5, color=color, linewidth=3)
        head_cov_lines.append(head_cov_line)
    true_pos = a1.plot(true_states[0, 0], true_states[0, 1], label="True Position", color='k', linewidth=2)
    true_pos2 = a2.plot(true_states[0, 0], true_states[0, 1], color='k', linewidth=2)
    full_pos2 = a2.plot(state_list[3][0, 0], state_list[3][0, 1], color='#eb3b5a', linewidth=2)
    d = a1.plot(true_states[0, 0], true_states[0, 1], 'o', color='k', markersize=10)[0]
    d2 = a2.plot(true_states[0, 0], true_states[0, 1], 'o', color='k', markersize=5)[0]
    
    def update(t):
        if t == 0: return
        for i in range(4):
            pos_lines[i][0].set_data(state_list[i][:t, 0], state_list[i][:t, 1])
            # pos_cov_lines[i][0].set_data(range(t), pos_cov_list[i][:t])
            head_cov_lines[i][0].set_data(range(t), head_cov_list[i][:t])
        true_pos[0].set_data(true_states[:t, 0], true_states[:t, 1])
        true_pos2[0].set_data(true_states[:t, 0], true_states[:t, 1])
        full_pos2[0].set_data(state_list[3][:t, 0], state_list[3][:t, 1])
        d.set_data(true_states[t-1, 0], true_states[t-1, 1])
        d2.set_data(true_states[t-1, 0], true_states[t-1, 1])

        mean, major, minor, angle = full_beliefs[t].marginalise(2).get_confidence_ellipse(1)
        arc.set_center((mean[0], mean[1]))
        arc.set_width(major)
        arc.set_height(minor)
        arc.set_angle(angle)


        pos = true_states[t-1, :2]
        buffer = 6
        x_min = max(pos[0] - buffer, -1)
        x_max = min(pos[0] + buffer, 29)
        y_min = max(pos[1] - buffer, 0)
        y_max = min(pos[1] + buffer, 27)
        a2.set_xlim(x_min, x_max)
        a2.set_ylim(y_min, y_max)

    a1.set_aspect('equal')
    a1.set_xlim(-1, 29)
    a1.set_ylim(0, 27)
    a2.set_aspect('equal')
    a3.set_xlim(0, 70)
    a3.set_ylim(0, 2)
    a1.grid(True)
    a2.grid(True)
    a3.grid(True)
    a1.set_title("Vehicle position")
    a2.set_ylabel("Position \nCovariance")
    a3.set_ylabel("Heading \nCovariance")
    a3.set_xlabel("Time")
    a3.set_xticklabels([f"{int(i/10)}" for i in a3.get_xticks()])
    # a2.set_xticklabels([f"{int(i/10)}" for i in a2.get_xticks()])
    a1.yaxis.set_major_locator(plt.MultipleLocator(5))
    fig.legend(loc="center", bbox_to_anchor=(0.25, 0.05), ncol=3)

    # plt.savefig(f"media/2D_car_{label}.svg")
    plt.tight_layout()

    # anim = FuncAnimation(fig, update, frames = 2, interval = 20) 
    # anim = FuncAnimation(fig, update, frames = f_control*3, interval = 20) 
    anim = FuncAnimation(fig, update, frames = simulation_time*f_control, interval = 20) 
    
    anim.save('media/SensorFusion_2D.gif', writer = 'ffmpeg', fps = 7) 




test_full_fusion()

