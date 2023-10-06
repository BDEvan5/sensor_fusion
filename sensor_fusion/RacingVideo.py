import numpy as np
from sensor_fusion.utils.Gaussian import Gaussian
from sensor_fusion.robots.AutonomousRacer import AutonomousRacer
from sensor_fusion.utils.utils import *

from sensor_fusion.filters.ParticleFilter import ParticleFilter
import cProfile, pstats, io

from matplotlib.animation import FuncAnimation  
from PIL import Image

def convert_to_map_coords(x, y, map_resolution=0.05, map_origin=[10, 22.6]):
    map_x = (x - map_origin[0]) / map_resolution
    map_y = (y - map_origin[1]) / map_resolution

    return map_x, map_y


def continuous_plot():
    Q = np.diag([0.1**2, 0.1**2, 0.1**2])
    R = np.diag([0.1**2])
    NP = 100

    f_s = 4
    T = 10
    init_state = np.array([0,0, 0])
    init_belief = Gaussian(init_state, np.diag([1**2, 1**2, 0.05**2]))
    robot = AutonomousRacer(init_state, 1/f_s)
    particle_filter = ParticleFilter(init_belief, robot.f, robot.h, Q, R, NP)

    fig = plt.figure(figsize=(10, 5))
    a1 = plt.subplot2grid((2, 4), (0, 0))
    a2 = plt.subplot2grid((2, 4), (0, 1))
    a3 = plt.subplot2grid((2, 4), (1, 0))
    a4 = plt.subplot2grid((2, 4), (1, 1))
    a5 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)

    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(top=0.9)

    plt.setp(a1, xticks=[], yticks=[])
    plt.setp(a2, xticks=[], yticks=[])
    plt.setp(a3, xticks=[], yticks=[])
    plt.setp(a4, xticks=[], yticks=[])
    plt.setp(a5, xticks=[], yticks=[])

    a1.set_title("Initial belief", fontweight="bold")
    a2.set_title("Motion update", fontweight="bold")
    a3.set_title("Measurement", fontweight="bold")
    a4.set_title("Resample", fontweight="bold")

    weight_scale = 4000
    length = 0.5
    block_size = 0.4
    map_block_size = 6
    map_length = 1 *20

    init_dist = a1.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#4b7bec', alpha=0.7)[0]
    init2 = a2.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#4b7bec', alpha=0.3, markersize=6)[0]
    # init4 = a4.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#4b7bec', alpha=0.1, markersize=6)[0]
    prior_dist = a2.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], 'o', color='#eb3b5a', alpha=0.9)[0]
    prior2 = a3.plot(particle_filter.particles[:, 0], particle_filter.particles[:, 1], '.', color='#eb3b5a', alpha=0.5)[0]
    weight_dist = a3.scatter(particle_filter.particles[:, 0], particle_filter.particles[:, 1], s=particle_filter.weights*weight_scale, color='#fd9644', alpha=0.9)
    weight4 = a4.scatter(particle_filter.particles[:, 0], particle_filter.particles[:, 1], s=particle_filter.weights*weight_scale, color='#fd9644', alpha=0.7)

    resample = a4.plot(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], '.', color='#20bf6b', alpha=0.99, markersize=8)[0]
    arrow = a4.arrow(0, 0, np.cos(0)*length, np.sin(0)*length, color='#a55eea', width=0.1, head_width=0.4, zorder=10, head_length=0.5)

    map_resolution = 0.05
    map_origin = [-10, -22.6]
    map_img_path = "maps/aut_wide.png"
    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    a5.imshow(map_img, cmap="gray", origin="lower")

    arrow2 = a5.arrow(0, 0, np.cos(0)*length, np.sin(0)*length, color='#a55eea', width=10, head_width=20, zorder=20, head_length=10)
    xs, ys = convert_to_map_coords(particle_filter.particles[:, 0], particle_filter.particles[:, 1], map_resolution, map_origin)
    init5 = a5.plot(xs, ys, 'o', color='#4b7bec', alpha=0.7, markersize=4)[0]
    xs, ys = convert_to_map_coords(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], map_resolution, map_origin)
    resample5 = a5.plot(xs, ys, 'o', color='#20bf6b', alpha=0.5)[0]

    beams = []
    for i in range(10):
        x1, y1 = convert_to_map_coords(0, 0, map_resolution, map_origin)
        beam_length = map_length * 20
        angle = i*(1/9)*np.pi-np.pi/2 
        x2, y2 = x1 + np.cos(angle)*beam_length, y1 + np.sin(angle)*beam_length
        b = a5.plot([x1, x2], [y1, y2], color='#0fb9b1', alpha=0.9, linewidth=2)[0]
        beams.append(b)

    # for k in range (1, 31):
    def update(k):
    # for k in range (1,T*f_s+1):

        init_dist.set_data(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1]) 
        init2.set_data(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1]) 
        xs, ys = convert_to_map_coords(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], map_resolution, map_origin)
        init5.set_data(xs, ys) 

        control = robot.move()
        particle_filter.control_update(control)
        for z in range(2):
            control = robot.move()
            particle_filter.particle_control_update(control)
        prior_dist.set_data(particle_filter.particles[:, 0], particle_filter.particles[:, 1])
        prior2.set_data(particle_filter.particles[:, 0], particle_filter.particles[:, 1])


        x_min = min(np.min(particle_filter.particles[:, 0]), np.min(particle_filter.proposal_distribution[:, 0])) - block_size
        x_max = max(np.max(particle_filter.particles[:, 0]), np.max(particle_filter.proposal_distribution[:, 0])) + block_size
        y_min = min(np.min(particle_filter.particles[:, 1]), np.min(particle_filter.proposal_distribution[:, 1])) - block_size
        y_max = max(np.max(particle_filter.particles[:, 1]), np.max(particle_filter.proposal_distribution[:, 1])) + block_size
        s_min = min(x_min, y_min)
        s_max = max(x_max, y_max)

        for a in (a1, a2):
            a.set_xlim(x_min, x_max)
            a.set_ylim(y_min, y_max)
            # a.set_aspect('equal', adjustable='box')

        measurement = robot.measure()
        particle_filter.measurement_update(measurement)
        states = np.array(robot.true_states)

        weight_dist.set_offsets(particle_filter.particles[:, :2])
        weight_dist.set_sizes(particle_filter.weights*weight_scale)
        weight4.set_offsets(particle_filter.particles[:, :2])
        weight4.set_sizes(particle_filter.weights*weight_scale)
        resample.set_data(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1])
        xs, ys = convert_to_map_coords(particle_filter.proposal_distribution[:, 0], particle_filter.proposal_distribution[:, 1], map_resolution, map_origin)
        resample5.set_data(xs, ys)
        arrow.set_data(x=states[-1, 0], y=states[-1, 1], dx=np.cos(states[-1, 2])*length, dy=np.sin(states[-1, 2])*length)
        x, y = convert_to_map_coords(states[-1, 0], states[-1, 1], map_resolution, map_origin)
        arrow2.set_data(x=x, y=y, dx=np.cos(states[-1, 2])*map_length, dy=np.sin(states[-1, 2])*map_length)

        for i in range(10):
            beam_length = measurement[i] * 20
            angle = i*(1/9)*np.pi-np.pi/2 + states[-1, 2]
            x2, y2 = x + np.cos(angle)*beam_length, y + np.sin(angle)*beam_length
            beams[i].set_data([x, x2], [y, y2])
        
        xs, ys = convert_to_map_coords(states[:, 0], states[:, 1], map_resolution, map_origin)
        a5.plot(xs, ys, color="k", label="True trajectory")

        x_min = min(np.min(particle_filter.particles[:, 0]), np.min(particle_filter.proposal_distribution[:, 0])) - block_size
        x_max = max(np.max(particle_filter.particles[:, 0]), np.max(particle_filter.proposal_distribution[:, 0])) + block_size
        y_min = min(np.min(particle_filter.particles[:, 1]), np.min(particle_filter.proposal_distribution[:, 1])) - block_size
        y_max = max(np.max(particle_filter.particles[:, 1]), np.max(particle_filter.proposal_distribution[:, 1])) + block_size

        for a in (a3, a4):
            a.set_xlim(x_min, x_max)
            a.set_ylim(y_min, y_max)
            # a.set_aspect('equal', adjustable='box')

        x_min, y_min = convert_to_map_coords(states[-1, 0] - map_block_size, states[-1, 1] - map_block_size, map_resolution, map_origin)
        x_max, y_max = convert_to_map_coords(states[-1, 0] + map_block_size, states[-1, 1] + map_block_size, map_resolution, map_origin)
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(map_img.shape[1], int(x_max))
        y_max = min(map_img.shape[0], int(y_max))
        a5.set_xlim(x_min, x_max)
        a5.set_ylim(y_min, y_max)

        return (init_dist, init2, prior_dist, prior2, weight_dist, weight4, resample, resample5, arrow, arrow2, init5, *beams)

        # plt.pause(0.5)
    # plt.show()



    anim = FuncAnimation(fig, update, frames = 42, interval = 20) 
    
    anim.save('media/ParticleFilter.gif', writer = 'ffmpeg', fps = 3) 



continuous_plot()

