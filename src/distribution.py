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
    def reweight(self, theta, Theta_x, size=None):
        if size is None:
            size = self.NUM_PARTICLES
        weights = np.array(self.weights)
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weights[i] *= pe.prob_theta_given_lam_stable(theta, particle, Theta_x, self.cost)
        try:
            w = weights / np.sum(weights)
        except:
            print "Exception: " + str(np.sum(weights))
        return weights / np.sum(weights)
    def resample(self, size=None):
        if size is None:
            size = self.NUM_PARTICLES
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
        for i in range(self.NUM_PARTICLES):
            particle = discretized[i]
            weight = self.weights[i]
            try:
                counts[tuple(particle)] += weight
            except:
                counts[tuple(particle)] = weight
        vals = np.array(counts.values()) / np.sum(counts.values())
        return -np.sum(vals * np.log2(vals / len(discretized)))
    def info_gain(self, Theta_x, box_size):
        new_weights = np.zeros(self.NUM_PARTICLES)
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            # for theta in Theta_x:
            #     weights = self.reweight(theta, Theta_x)
            #     weights *= weight * pe.prob_theta_given_lam_stable2(theta, particle, Theta_x, self.cost, 100)
            #     new_weights += weights
            theta = pe.mle(Theta_x, particle, self.cost, lambda x: 1)
            weights = self.reweight(theta, Theta_x)
            weights *= weight
            new_weights += weights
        dist = ParticleDistribution(self.particles, new_weights, self.cost)
        # print self.weights
        # print new_weights
        return dist.entropy(box_size=box_size) - self.entropy(box_size=box_size)

