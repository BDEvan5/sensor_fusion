from matplotlib import pyplot as plt
import numpy as np
from matplotlib import patches as pat

def plot_estimation(robot, filter):
    true_states, dr_states = robot.get_states()
    estimates = np.array(filter.get_estimated_states())
    dr_mae = np.mean(np.linalg.norm(true_states - dr_states, axis=1), axis=0)
    mae = np.mean(np.linalg.norm(true_states - estimates, axis=1), axis=0)
    print(f"Deadreckon MAE: {dr_mae:.4f} m --> Estimated MAE: {mae:.4f} m")

    plt.figure(figsize=(8, 5))
    plt.plot(true_states[:,0], true_states[:,1], color="k", label="True states")
    plt.plot(dr_states[:,0], dr_states[:,1], color="b", label="Dead reckoning")
    plt.plot(estimates[:,0], estimates[:,1], color="r", label="Estimated states")
    
    measurements = np.array(robot.measurements)
    for i, m in enumerate(measurements):
        plt.plot(robot.beacon[0] + m[0]*np.cos(m[1] + true_states[i, 2]), robot.beacon[1]+ m[0]*np.sin(m[1] + true_states[i, 2]), 'go', alpha=0.5)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(filter.name)

    plt.savefig(f"media/{filter.name}.svg")

    plt.show()

def plot_estimation_belief(robot, filter):
    true_states, dr_states = robot.get_states()
    estimates = np.array(filter.get_estimated_states())
    dr_mae = np.mean(np.linalg.norm(true_states - dr_states, axis=1), axis=0)
    mae = np.mean(np.linalg.norm(true_states - estimates, axis=1), axis=0)
    print(f"Deadreckon MAE: {dr_mae:.4f} m --> Estimated MAE: {mae:.4f} m")

    plt.figure(figsize=(8, 5))
    plt.plot(true_states[:,0], true_states[:,1], color="k", label="True states")
    plt.plot(dr_states[:,0], dr_states[:,1], color="b", label="Dead reckoning")
    plt.plot(estimates[:,0], estimates[:,1], color="r", label="Estimated states")

    measurements = np.array(robot.measurements)
    for i, m in enumerate(measurements):
        plt.plot(robot.beacon[0] + m[0]*np.cos(m[1] + true_states[i, 2]), robot.beacon[1]+ m[0]*np.sin(m[1] + true_states[i, 2]), 'go', alpha=0.5)

    beliefs = filter.beliefs
    for belief in beliefs:
        mean, major, minor, angle = belief.marginalise(2).get_confidence_ellipse(1)
        arc = pat.Arc((mean[0],mean[1]),major,minor,angle,color="0.5")
        plt.gca().add_patch(arc)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(filter.name)

    plt.savefig(f"media/{filter.name}.svg")

    plt.show()


