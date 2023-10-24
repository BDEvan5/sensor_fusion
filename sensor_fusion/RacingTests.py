import numpy as np
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.AutonomousRacer import AutonomousRacer
from sensor_fusion.utils.utils import *

from sensor_fusion.estimators.ParticleFilter import ParticleFilter
import cProfile, pstats, io


def simulate_racing(robot, estimator, f_s, T):
    for k in range (1,T*f_s+1):
        control = robot.move()
        estimator.control_update(control)
        measurement = robot.measure()
        if measurement is not None:
            estimator.measurement_update(measurement)

def simulate_car_pf():
    Q = np.diag([0.1**2, 0.1**2, 0.05**2])
    R = np.diag([0.1**2])
    NP = 50

    f_s = 5
    init_state = np.array([0,0, 0])
    init_belief = Gaussian(init_state, np.diag([2**2,2**2,0.1**2]))
    robot = AutonomousRacer(init_state, 1/f_s)
    particle_filter = ParticleFilter(init_belief, robot.f, robot.h, Q, R, NP)
    
    simulate_racing(robot, particle_filter, f_s, 31)

    true_states, noisy_states = robot.get_states()
    estimates = particle_filter.get_estimated_states()
    plot_racing_localistion(estimates, true_states, noisy_states, robot)

    mae = np.mean(np.abs(true_states - estimates))
    print(f"MAE: {mae:.4f} cm")


def profile_particle_filter(number_of_particles):
    Q = np.diag([0.1**2, 0.1**2, 0.05**2])
    R = np.diag([0.1**2])

    f_s = 5
    init_state = np.array([0,0, 0])
    init_belief = Gaussian(init_state, np.diag([0.8**2, 0.8**2, 0.05**2]))

    def measure_compute(NP):
        robot = AutonomousRacer(init_state, 1/f_s)
        particle_filter = ParticleFilter(init_belief, robot.f, robot.h, Q, R, number_of_particles)
        
        simulate_racing(robot, particle_filter, f_s, 10)

        true_states, _noisy_states = robot.get_states()
        return true_states, particle_filter.get_estimated_states()

    pr = cProfile.Profile()
    pr.enable()
    true_states, estimates = measure_compute(np)
    pr.disable()
    
    ps = pstats.Stats(pr).strip_dirs().sort_stats('cumulative')
    ps.print_stats('measurement_update')

    print(f"Mean absolute error: {np.mean(np.abs(true_states - estimates)):.4f} cm")

def continuous_plot():
    Q = np.diag([0.4**2, 0.4**2, 0.05**2])
    R = np.diag([0.1**2])
    NP = 800

    f_s = 4
    T = 10
    init_state = np.array([0,0, 0])
    init_belief = Gaussian(init_state, np.diag([1**2, 1**2, 0.05**2]))
    robot = AutonomousRacer(init_state, 1/f_s)
    particle_filter = ParticleFilter(init_belief, robot.f, robot.h, Q, R, NP)

    plt.figure(figsize=(8, 8))
    for k in range (1,T*f_s+1):
        plt.clf()

        plt.scatter(particle_filter.particles[:, 0], particle_filter.particles[:, 1], color='b', label="Proposal distribution", alpha=0.5)

        control = robot.move()
        particle_filter.control_update(control)
        plt.scatter(particle_filter.particles[:, 0], particle_filter.particles[:, 1], color='g', label="Prior distribution", alpha=0.5)

        measurement = robot.measure()
        particle_filter.measurement_update(measurement)
        plt.scatter(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], color='r', label="Resampled distribution", alpha=0.5)

        states = np.array(robot.true_states)
        plt.plot(states[:,0], states[:,1], color="k", label="True trajectory")
        plt.legend()
        s = 2
        plt.xlim(states[-1, 0] - s, states[-1, 0] + s)
        plt.ylim(states[-1, 1] - s, states[-1, 1] + s)

        plt.pause(0.5)
        # plt.show()

    # true_states, noisy_states = robot.get_states()
    # estimates = particle_filter.get_estimated_states()
    # make_estimation_plot(estimates, true_states, noisy_states, robot)

    # mae = np.mean(np.abs(true_states - estimates))
    # print(f"MAE: {mae:.4f} cm")




def plot_racing_localistion(estimates, true_states, noisy_states, robot):
    plt.figure(figsize=(8, 8))
    map_img = robot.scan_simulator.map_img
    map_img[map_img == 0] = 100
    map_img[0, 0] = 0
    plt.imshow(map_img, cmap="gray", origin="lower")
    true_states = robot.scan_simulator.xy_2_rc(true_states)
    noisy_states = robot.scan_simulator.xy_2_rc(noisy_states)
    estimates = robot.scan_simulator.xy_2_rc(estimates)
    
    plt.plot(true_states[:,0],true_states[:,1],color="k", label=f"True trajectory")
    plt.plot(noisy_states[:,0],noisy_states[:,1],color="b", label="Dead rekoning")
    plt.plot(estimates[:,0],estimates[:,1], 'x-', color="r", label="Estimated PF")

    plt.legend()

    plt.savefig("media/Particle Filter localisation.svg")
    plt.show()


simulate_car_pf()
# continuous_plot()

# profile_particle_filter(5)
