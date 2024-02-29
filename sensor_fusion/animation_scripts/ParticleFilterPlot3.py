import numpy as np
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.AutonomousRacer import AutonomousRacer
from sensor_fusion.utils.utils import *
from sensor_fusion.estimators.ParticleFilter import ParticleFilter
from matplotlib import cm 
from PIL import Image

def convert_to_map_coords(x, y, map_resolution=0.05, map_origin=[10, 22.6]):
    map_x = (x - map_origin[0]) / map_resolution
    map_y = (y - map_origin[1]) / map_resolution

    return map_x, map_y

weight_scale = 4000
length = 0.3
block_size = 0.4
map_block_size = 6
map_length = 1 *20


map_resolution = 0.05
map_origin = [-10, -22.6]
np.random.seed(42)

def make_animation():
    Q = np.diag([0.1**2, 0.1**2, 0.1**2])
    R = np.diag([0.1**2])
    NP = 20

    f_s = 4
    init_state = np.array([-3, -7.2, 2.7])
    # init_state = np.array([-8, -7, 1.7])
    # init_state = np.array([10, -12, 1.7])
    # init_state = np.array([9, -14, 0.7])
    # init_state = np.array([0,0, 0])
    init_belief = Gaussian(init_state, np.diag([0.5**2, 0.5**2, 0.26**2]))
    # init_belief = Gaussian(init_state, np.diag([1**2, 1**2, 0.05**2]))
    robot = AutonomousRacer(init_state, 1/f_s)
    particle_filter = ParticleFilter(init_belief, robot.f, robot.h, Q, R, NP)

    fig = plt.figure(figsize=(6, 2))
    a2 = plt.subplot2grid((1, 3), (0, 0))
    a4 = plt.subplot2grid((1, 3), (0, 1))
    a5 = plt.subplot2grid((1, 3), (0, 2))

    for a in (a2, a4, a5): plt.setp(a, xticks=[], yticks=[])


    map_img_path = "maps/aut_wide.png"
    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img[map_img > 128.] = 180
    map_img[map_img <= 128.] = 230
    map_img[0, 1] = 255
    map_img[0, 0] = 0
    a5.imshow(map_img, cmap="gray", origin="lower")

    # plot initial distributions
    a2.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#4b7bec', alpha=0.7, markersize=6, label="Initial belief")[0]
    # a4.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#4b7bec', alpha=0.3, markersize=6)[0]
    xs, ys = convert_to_map_coords(particle_filter.particles[:, 0], particle_filter.particles[:, 1], map_resolution, map_origin)
    a5.plot(xs, ys, 'o', color='#4b7bec', alpha=0.7, markersize=4)[0]

    # update pf
    control = robot.move()
    particle_filter.control_update(control)
    for z in range(2):
        control = robot.move()
        particle_filter.control_update(control, False)

    # plot movement update
    a2.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#eb3b5a', alpha=0.9, label="Motion update")[0]
    a2.legend(loc='lower left', fontsize=9)

    x_min = min(np.min(particle_filter.particles[:, 0]), np.min(particle_filter.proposal_distribution[:, 0])) - block_size
    x_max = max(np.max(particle_filter.particles[:, 0]), np.max(particle_filter.proposal_distribution[:, 0])) + block_size
    y_min = min(np.min(particle_filter.particles[:, 1]), np.min(particle_filter.proposal_distribution[:, 1])) - block_size
    y_max = max(np.max(particle_filter.particles[:, 1]), np.max(particle_filter.proposal_distribution[:, 1])) + block_size

    a2.set_xlim(x_min, x_max)
    a2.set_ylim(y_min-0.5, y_max)
    a2.text(0.1, 0.9, "1", fontdict={"fontweight": "bold"}, fontsize=14, transform=a2.transAxes, ha="center", va="center")
    a4.text(0.1, 0.9, "2", fontdict={"fontweight": "bold"}, fontsize=14, transform=a4.transAxes, ha="center", va="center")

    measurement = robot.measure()
    particle_filter.measurement_update(measurement)
    states = np.array(robot.true_states)

    # plot measurement update
    a4.scatter(particle_filter.particles[:, 0], particle_filter.particles[:, 1], s=particle_filter.weights*weight_scale, color='#eb3b5a', alpha=0.6, label="Particle weights")

    leg = a4.legend(fontsize=9, loc='lower left')
    leg.legendHandles[0]._sizes = [80]
    a4.arrow(x=states[-1, 0], y=states[-1, 1], dx=np.cos(states[-1, 2])*length, dy=np.sin(states[-1, 2])*length, color='#a55eea', width=0.16, head_width=0.32, zorder=10, head_length=0.22)

    xs, ys = convert_to_map_coords(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], map_resolution, map_origin)

    x, y = convert_to_map_coords(states[-1, 0], states[-1, 1], map_resolution, map_origin)
    a5.arrow(x=x, y=y, dx=np.cos(states[-1, 2])*map_length, dy=np.sin(states[-1, 2])*map_length, color='#a55eea', width=10, head_width=20, zorder=20, head_length=15, label='Estimated pose')

    anlges = np.linspace(-np.pi/2, np.pi/2, len(measurement))+ states[-1, 2]
    xs = np.cos(anlges)*20*measurement + x
    ys = np.sin(anlges)*20*measurement + y
    a5.scatter(xs, ys, c=measurement, cmap=cm.get_cmap("gist_rainbow"))
    
    xs, ys = convert_to_map_coords(states[:, 0], states[:, 1], map_resolution, map_origin)
    a5.plot(xs, ys, color="k")

    block_x = 0.45
    block_y = -0.
    x_min = min(np.min(particle_filter.particles[:, 0]), np.min(particle_filter.proposal_distribution[:, 0])) - block_x
    x_max = max(np.max(particle_filter.particles[:, 0]), np.max(particle_filter.proposal_distribution[:, 0])) + block_x
    y_min = min(np.min(particle_filter.particles[:, 1]), np.min(particle_filter.proposal_distribution[:, 1])) - block_y
    y_max = max(np.max(particle_filter.particles[:, 1]), np.max(particle_filter.proposal_distribution[:, 1])) + block_y

    a4.set_xlim(x_min, x_max)
    a4.set_ylim(y_min, y_max)

    a2.set_aspect('equal', adjustable='box')
    a4.set_aspect('equal', adjustable='box')

    map_block_size_x = 4
    map_block_size_y = 6
    x_min, y_min = convert_to_map_coords(states[-1, 0] - map_block_size_x-2, states[-1, 1] - map_block_size_y+2.5, map_resolution, map_origin)
    x_max, y_max = convert_to_map_coords(states[-1, 0] + map_block_size_x, states[-1, 1] + map_block_size_y-0.4, map_resolution, map_origin)
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(map_img.shape[1], int(x_max))
    y_max = min(map_img.shape[0], int(y_max))
    a5.set_xlim(x_min, x_max)
    a5.set_ylim(y_min, y_max)

    plt.tight_layout(pad=0.1) # position is important
    # plt.tight_layout(pad=0.5) # position is important

    a5.legend(loc='lower left', fontsize=9)
    # a5.legend(loc='upper center', bbox_to_anchor=(0.5, 0.02))

    pos = a5.get_position()
    print(pos)
    # a5.set_position([0.67, 0.3, pos.x1 - pos.x0, pos.y1 - pos.y0])
    pos = a5.get_position()
    print(pos)

    # plt.subplots_adjust(bottom=0.01)

    plt.savefig(f"media/particle_filter_plot3.svg")
    plt.savefig(f"media/particle_filter_plot3.pdf")



make_animation()

