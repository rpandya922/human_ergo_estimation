from __future__ import division
import numpy as np
from scipy.misc import logsumexp
import probability_estimation as pe

h = 0.2
class ParticleDistribution():
    def __init__(self, particles, weights, cost):
        self.NUM_PARTICLES = len(particles)
        self.particles = particles
        self.weights = weights
        self.cost = cost
    def resample(self, theta, Theta_x, size=None):
        if size is None:
            size = self.NUM_PARTICLES
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            w = particle[:3]
            theta_star = particle[3:]
            log_likelihood = np.log(self.weights[i])
            p, costs = pe.prob_theta_given_lam_stable(theta, theta_star, w, Theta_x, self.cost)
            log_likelihood += p - logsumexp(costs)
            self.weights[i] = log_likelihood
        self.weights /= np.sum(self.weights)
        new_particles = []
        idxs = np.random.choice(self.NUM_PARTICLES, size=size, p=self.weights)
        for i in range(size):
            idx = idxs[i]
            sample = self.particles[idx]
            new_particles.append(np.random.normal(sample, h))
        self.particles = new_particles
        self.weights = [1/size]*size
        self.NUM_PARTICLES = size
    def entropy(self, box_size=0.001):
        discretized = np.floor(np.array(self.particles) / box_size) * box_size
        counts = {}
        for particle in discretized:
            try:
                counts[tuple(particle)] += 1
            except:
                counts[tuple(particle)] = 1
        vals = np.array(counts.values())
        return -np.sum(vals * np.log2(vals / len(discretized)))
