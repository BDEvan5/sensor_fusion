import numpy as np
import matplotlib.pyplot as plt

def plot_linear_fusion(car, estimator, label):
    plt.figure(figsize=(10, 6))
    a1 = plt.subplot(2, 2, 1)
    a2 = plt.subplot(2, 2, 3)
    a3 = plt.subplot(2, 2, 2)
    a4 = plt.subplot(2, 2, 4)

    states = estimator.get_estimated_states()
    covariances = estimator.get_estimated_covariances()

    a1.plot(states[:, 0], label=label)
    a2.plot(states[:, 1])
    a3.plot(covariances[:, 0, 0]**0.5, label=label)
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



def plot_estimation_comparison(car, filter_list, label_list, save_name):
    plt.figure(figsize=(12, 8))
    a1 = plt.subplot(2, 2, 1)
    a2 = plt.subplot(2, 2, 3)
    a3 = plt.subplot(2, 2, 2)
    a4 = plt.subplot(2, 2, 4)

    for f, filter in enumerate(filter_list):
        states = filter.get_estimated_states()
        covariances = filter.get_estimated_covariances()

        a1.plot(states[:, 0], label=label_list[f])
        a2.plot(states[:, 1])
        a3.plot(covariances[:, 0, 0]**0.5, label=label_list[f])
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



def plot_2D_comparison(estimator_list, car):
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


