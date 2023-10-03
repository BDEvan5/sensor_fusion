import numpy as np


class ParticleFilter:
    def __init__(self, init_belief, f, h, Q, R, NP) -> None:
        self.estimates = [init_belief.get_mean()]
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.NP = NP

        self.particles = init_belief.draw_samples(self.NP)
        self.proposal_distribution = self.particles 
        self.weights = np.ones(self.NP) / self.NP
        self.particle_indices = np.arange(self.NP)

    def control_update(self, control):
        next_states = self.f(self.proposal_distribution, control)
        random_samples = np.random.multivariate_normal(np.zeros(3), self.Q, self.NP)
        self.particles = next_states + random_samples

    def measurement_update(self, measurement):
        particle_measurements = self.h(self.particles)
        z = particle_measurements - measurement
        sigma = np.sqrt(np.average(z**2, axis=0))
        weights = 1.0 / np.sqrt(2.0 * np.pi * sigma ** 2) * np.exp(-z ** 2 / (2 * sigma ** 2))
        self.weights = np.prod(weights, axis=1) 

        weight_sum = np.sum(self.weights, axis=0)
        if (weight_sum == 0).any():
            print(f"Problem with weights")

        self.weights = self.weights / np.sum(self.weights)
        estimate = np.dot(self.particles.T, self.weights)
        self.estimates.append(estimate)

        proposal_indices = np.random.choice(self.particle_indices, self.NP, p=self.weights)
        self.proposal_distribution = self.particles[proposal_indices,:]

    def get_estimated_states(self):
        return np.array(self.estimates)




