from __future__ import division
import numpy as np
from scipy.misc import logsumexp
import probability_estimation as pe

h = 0.5
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
            # new_particles.append(sample)
        self.particles = new_particles
        self.weights = [1/size]*size
        self.NUM_PARTICLES = size
    def entropy(self, num_boxes=10, axis_ranges=None):
        if axis_ranges is None:
            axis_ranges = [20]*len(self.particles[0])
        box_sizes = axis_ranges / num_boxes
        discretized = np.round(np.array(self.particles) / box_sizes) * box_sizes
        counts = {}
        for i in range(self.NUM_PARTICLES):
            particle = discretized[i]
            weight = self.weights[i]
            try:
                counts[tuple(particle)] += weight
            except:
                counts[tuple(particle)] = weight
        # bins = len(counts.keys())
        # print "Number of bins: " + str(bins)
        # maxent = -np.log2(1 / bins)
        # print "Max possible entropy: " + str(maxent)
        vals = np.array(counts.values()) / np.sum(counts.values())
        return -np.sum(vals * np.log2(vals))
    def info_gain(self, Theta_x, num_boxes=10, axis_ranges=None):
        new_weights = np.zeros(self.NUM_PARTICLES)
        avg_ent = 0
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            for theta in Theta_x:
                weights = self.reweight(theta, Theta_x)
                d = ParticleDistribution(self.particles, weights, self.cost)
                d.resample()
                ent = d.entropy(num_boxes, axis_ranges)
                avg_ent += weight * pe.prob_theta_given_lam_stable2(theta, particle, Theta_x, self.cost, 1) * ent
                # weights *= weight * pe.prob_theta_given_lam_stable2(theta, particle, Theta_x, self.cost, 1)
                # new_weights += weights
            # theta = pe.mle(Theta_x, particle, self.cost, lambda x: 1)
            # weights = self.reweight(theta, Theta_x)
            # d = ParticleDistribution(self.particles, weights, self.cost)
            # d.resample()
            # ent = d.entropy(num_boxes, axis_ranges)
            # avg_ent += weight * pe.prob_theta_given_lam_stable2(theta, particle, Theta_x, self.cost, 1) * ent
            # weights *= weight
            # new_weights += weights
        return avg_ent - self.entropy(num_boxes, axis_ranges)
        dist = ParticleDistribution(self.particles, new_weights, self.cost)
        dist.resample()
        # print self.weights
        # print new_weights
        return dist.entropy(num_boxes, axis_ranges) - self.entropy(num_boxes, axis_ranges)

