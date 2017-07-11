from __future__ import division
import numpy as np
from scipy.misc import logsumexp
from scipy.stats import gaussian_kde as kde
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
    def reweight_general(self, observation, prob):
        weights = np.array(self.weights)
        mult = np.array([prob(observation, particle) for particle in self.particles])
        weights *= mult
        self.weights = weights / np.sum(weights)
    def plot_kde(self, ax1, ax2):
        ws = np.zeros(self.NUM_PARTICLES)
        means = np.zeros(self.NUM_PARTICLES)
        for i, particle in enumerate(self.particles):
            ws[i] = particle[0]
            means[i] = particle[1]
        kernel1 = kde(ws)
        kernel2 = kde(means)
        x1 = np.linspace(-5, 10, 2000)
        x2 = np.linspace(-6, 6, 2000)
        pdf1 = kernel1.pdf(x1)
        pdf2 = kernel2.pdf(x2)
        mode1 = x1[np.argmax(pdf1)]
        mode2 = x2[np.argmax(pdf2)]
        ax1.set_title('weights')
        ax2.set_title('means')
        ax1.set_ylim(0, 1.5)
        ax2.set_ylim(0, 1.5)
        ax1.plot(x1, pdf1, label='mode=' + str(mode1))
        ax1.hist(ws, normed=True)
        ax2.plot(x2, pdf2, label='mode=' + str(mode2))
        ax2.hist(means, normed=True)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
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

class SetWeightsParticleDistribution():
    def __init__(self, particles, weights, cost, w, ALPHA_I, ALPHA_O):
        self.NUM_PARTICLES = len(particles)
        self.particles = particles
        self.weights = weights
        self.cost = cost
        self.w = w
        self.ALPHA_I = ALPHA_I
        self.ALPHA_O = ALPHA_O
    def reweight(self, theta, feasible):
        weights = np.array(self.weights)
        mult = np.zeros(self.NUM_PARTICLES)
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            if particle[0] < np.amin(feasible) or particle[0] > np.amax(feasible):
                alpha = self.ALPHA_O
            else:
                alpha = self.ALPHA_I
            mult[i] = pe.prob_theta_given_lam_stable_set_weight_num(theta, particle, self.w, self.cost, alpha)
            mult[i] -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, particle, self.w, self.cost, alpha)
        mult = np.exp(mult)
        weights *= mult
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
            axis_ranges = np.array([20]*len(self.particles[0]))
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
        vals = np.array(counts.values()) / np.sum(counts.values())
        return -np.sum(vals * np.log2(vals))
    def info_gain(self, feasible, num_boxes=10, axis_ranges=None):
        new_weights = np.zeros(self.NUM_PARTICLES)
        avg_ent = 0
        alpha = self.ALPHA_I
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            theta = max(feasible, key=lambda x: pe.prob_theta_given_lam_stable_set_weight_num(x, particle, self.w, self.cost, alpha))
            weights = self.reweight(theta, feasible)
            d = SetWeightsParticleDistribution(self.particles, weights, self.cost, self.w, self.ALPHA_I, self.ALPHA_O)
            ent = d.entropy(num_boxes, axis_ranges)
            avg_ent += weight * ent
            print i
        return self.entropy(num_boxes, axis_ranges) - avg_ent
    def discretize_to_probs(self, feasible, num_boxes=10, axis_ranges=None):
        if axis_ranges is None:
            axis_ranges = np.array([20]*len(self.particles[0]))
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
        return np.array(counts.values()) / np.sum(counts.values())
    def info_gain_kl(self, feasible, num_boxes=10, axis_ranges=None):
        ps = self.discretize_to_probs(feasible, num_boxes, axis_ranges)
