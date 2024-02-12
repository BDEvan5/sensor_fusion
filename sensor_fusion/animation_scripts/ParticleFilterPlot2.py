import numpy as np
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.AutonomousRacer import AutonomousRacer
from sensor_fusion.utils.utils import *
from sensor_fusion.estimators.ParticleFilter import ParticleFilter

from PIL import Image

def convert_to_map_coords(x, y, map_resolution=0.05, map_origin=[10, 22.6]):
    map_x = (x - map_origin[0]) / map_resolution
    map_y = (y - map_origin[1]) / map_resolution

    return map_x, map_y

weight_scale = 4000
length = 0.5
block_size = 0.4
map_block_size = 6
map_length = 1 *20


map_resolution = 0.05
map_origin = [-10, -22.6]
np.random.seed(42)

def make_animation():
    Q = np.diag([0.1**2, 0.1**2, 0.1**2])
    R = np.diag([0.1**2])
    NP = 100

    f_s = 4
    init_state = np.array([10, -12, 1.7])
    # init_state = np.array([9, -14, 0.7])
    # init_state = np.array([0,0, 0])
    init_belief = Gaussian(init_state, np.diag([0.5**2, 0.8**2, 0.05**2]))
    # init_belief = Gaussian(init_state, np.diag([1**2, 1**2, 0.05**2]))
    robot = AutonomousRacer(init_state, 1/f_s)
    particle_filter = ParticleFilter(init_belief, robot.f, robot.h, Q, R, NP)

    fig = plt.figure(figsize=(6.2, 3.))
    # fig = plt.figure(figsize=(7, 4))
    # fig = plt.figure(figsize=(9, 5))
    a1 = plt.subplot2grid((2, 3), (0, 0))
    a2 = plt.subplot2grid((2, 3), (0, 1))
    a3 = plt.subplot2grid((2, 3), (1, 0))
    a4 = plt.subplot2grid((2, 3), (1, 1))
    a5 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=2)
    # a1 = plt.subplot2grid((2, 4), (0, 0))
    # a2 = plt.subplot2grid((2, 4), (0, 1))
    # a3 = plt.subplot2grid((2, 4), (1, 0))
    # a4 = plt.subplot2grid((2, 4), (1, 1))
    # a5 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2, )

    # plt.tight_layout(pad=1.5)
    # plt.subplots_adjust(top=0.95)
    # plt.subplots_adjust(hspace=0.01)

    for a in (a1, a2, a3, a4, a5): plt.setp(a, xticks=[], yticks=[])

    # a1.set_title("Initial belief", fontweight="bold")
    # a2.set_title("Motion update", fontweight="bold")
    # a3.set_title("Measurement", fontweight="bold")
    # a4.set_title("Resample", fontweight="bold")

    map_img_path = "maps/aut_wide.png"
    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    
    a5.imshow(map_img, cmap="gray", origin="lower")

    # plot initial distributions
    a1.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#4b7bec', alpha=0.7)[0]
    a1.legend(["Initial belief"])
    a2.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#4b7bec', alpha=0.3, markersize=6)[0]
    xs, ys = convert_to_map_coords(particle_filter.particles[:, 0], particle_filter.particles[:, 1], map_resolution, map_origin)
    a5.plot(xs, ys, 'o', color='#4b7bec', alpha=0.7, markersize=4, label="Initial belief")[0]

    # update pf
    control = robot.move()
    particle_filter.control_update(control)
    for z in range(3):
        control = robot.move()
        particle_filter.control_update(control, False)

    # plot movement update
    a2.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#eb3b5a', alpha=0.9, label="Motion update")[0]
    a2.legend()
    a3.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], '.', color='#eb3b5a', alpha=0.5)[0]

    x_min = min(np.min(particle_filter.particles[:, 0]), np.min(particle_filter.proposal_distribution[:, 0])) - block_size
    x_max = max(np.max(particle_filter.particles[:, 0]), np.max(particle_filter.proposal_distribution[:, 0])) + block_size
    y_min = min(np.min(particle_filter.particles[:, 1]), np.min(particle_filter.proposal_distribution[:, 1])) - block_size
    y_max = max(np.max(particle_filter.particles[:, 1]), np.max(particle_filter.proposal_distribution[:, 1])) + block_size

    txt_x = 0.3
    txt_y = 1.1
    for a in (a1, a2):
        a.set_xlim(x_min, x_max)
        a.set_ylim(y_min, y_max)
        # a.set_aspect('equal', adjustable='box')
    a1.text(x_min+txt_x, y_max-txt_y, "1", fontdict={"fontweight": "bold"}, fontsize=14)
    a2.text(x_min+txt_x, y_max-txt_y, "2", fontdict={"fontweight": "bold"}, fontsize=14)

    measurement = robot.measure()
    particle_filter.measurement_update(measurement)
    states = np.array(robot.true_states)

    # plot measurement update
    a3.scatter(particle_filter.particles[:, 0], particle_filter.particles[:, 1], s=particle_filter.weights*weight_scale, color='#fd9644', alpha=0.9, label="Measurement \nweights")
    a3.legend()
    a4.scatter(particle_filter.particles[:, 0], particle_filter.particles[:, 1], s=particle_filter.weights*weight_scale, color='#fd9644', alpha=0.7)

    a4.plot(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], '.', color='#20bf6b', alpha=0.99, markersize=8, label="Resampled \nparticles")[0]
    a4.legend()
    a4.arrow(x=states[-1, 0], y=states[-1, 1], dx=np.cos(states[-1, 2])*length, dy=np.sin(states[-1, 2])*length, color='#a55eea', width=0.1, head_width=0.4, zorder=10, head_length=0.5)

    xs, ys = convert_to_map_coords(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], map_resolution, map_origin)
    a5.plot(xs, ys, 'o', color='#20bf6b', alpha=0.5, label="Final belief")[0]

    x, y = convert_to_map_coords(states[-1, 0], states[-1, 1], map_resolution, map_origin)
    a5.arrow(x=x, y=y, dx=np.cos(states[-1, 2])*map_length, dy=np.sin(states[-1, 2])*map_length, color='#a55eea', width=10, head_width=20, zorder=20, head_length=10)
    for i in range(10):
        beam_length = measurement[i] * 20
        angle = i*(1/9)*np.pi-np.pi/2 + states[-1, 2]
        x2, y2 = x + np.cos(angle)*beam_length, y + np.sin(angle)*beam_length
        if i == 0:
            a5.plot([x, x2], [y, y2], color='#0fb9b1', alpha=0.9, linewidth=2, label="LiDAR beams")[0]
        else:
            a5.plot([x, x2], [y, y2], color='#0fb9b1', alpha=0.9, linewidth=2)[0]
    
    xs, ys = convert_to_map_coords(states[:, 0], states[:, 1], map_resolution, map_origin)
    # a5.plot(xs, ys, color="k")
    a5.plot(xs, ys, color="k", label="Vehicle path")

    x_min = min(np.min(particle_filter.particles[:, 0]), np.min(particle_filter.proposal_distribution[:, 0])) - block_size
    x_max = max(np.max(particle_filter.particles[:, 0]), np.max(particle_filter.proposal_distribution[:, 0])) + block_size
    y_min = min(np.min(particle_filter.particles[:, 1]), np.min(particle_filter.proposal_distribution[:, 1])) - block_size
    y_max = max(np.max(particle_filter.particles[:, 1]), np.max(particle_filter.proposal_distribution[:, 1])) + block_size

    for a in (a3, a4):
        a.set_xlim(x_min, x_max)
        a.set_ylim(y_min, y_max)
        # a.set_aspect('equal', adjustable='box')

    txt_x = 0.2
    txt_y = 0.7
    a3.text(x_min+txt_x, y_max-txt_y, "3", fontdict={"fontweight": "bold"}, fontsize=14)
    a4.text(x_min+txt_x, y_max-txt_y, "4", fontdict={"fontweight": "bold"}, fontsize=14)

    map_block_size_x = 4
    map_block_size_y = 5
    x_min, y_min = convert_to_map_coords(states[-1, 0] - map_block_size_x, states[-1, 1] - map_block_size_y+0.2, map_resolution, map_origin)
    x_max, y_max = convert_to_map_coords(states[-1, 0] + map_block_size_x, states[-1, 1] + map_block_size_y-1, map_resolution, map_origin)
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(map_img.shape[1], int(x_max))
    y_max = min(map_img.shape[0], int(y_max))
    a5.set_xlim(x_min, x_max)
    a5.set_ylim(y_min, y_max)

    plt.tight_layout(pad=0.5) # position is important

    a5.legend(loc='upper center', bbox_to_anchor=(0.5, 0.02))

    pos = a5.get_position()
    print(pos)
    a5.set_position([0.67, 0.3, pos.x1 - pos.x0, pos.y1 - pos.y0])
    pos = a5.get_position()
    print(pos)

    # plt.subplots_adjust(bottom=0.01)

    plt.savefig(f"media/particle_filter_plot.svg")
    plt.savefig(f"media/particle_filter_plot.pdf")



make_animation()

