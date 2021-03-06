from __future__ import division
import numpy as np
from scipy.misc import logsumexp
from scipy.stats import gaussian_kde as kde
from scipy.optimize import minimize
import probability_estimation as pe
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal as mvn
import seaborn
import matplotlib.pyplot as plt
import sys

H = 0.03
# H = 0.2
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
    def __init__(self, particles, weights, cost, w, ALPHA_I, ALPHA_O, h=None):
        self.NUM_PARTICLES = len(particles)
        self.particles = particles
        self.weights = weights
        self.cost = cost
        self.w = w
        self.ALPHA_I = ALPHA_I
        self.ALPHA_O = ALPHA_O
        if h is None:
            self.h = H
        else:
            self.h = h
    def reweight_vectorized(self, theta, feasible):
        weights = np.array(self.weights)
        particles = np.array(self.particles)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
        distances, indices = nbrs.kneighbors(feasible)
        max_dist = np.amax(distances)
        distances, indices = nbrs.kneighbors(self.particles)
        is_outside = distances[:,0] >= max_dist
        is_inside = ~ is_outside

        c = self.cost(theta, particles, self.w)
        costs_inside = -self.ALPHA_I * c
        costs_outside = -self.ALPHA_O * c

        feas_tiled = np.repeat(feasible[:,:,np.newaxis], self.NUM_PARTICLES, axis=2)
        theta_stars = particles.T
        d_theta = np.square(feas_tiled - theta_stars)
        costs_denom = np.swapaxes(d_theta, 1, 2).dot(self.w)

        costs_denom_inside = logsumexp(-self.ALPHA_I * costs_denom, axis=0)
        costs_denom_outside = logsumexp(-self.ALPHA_O * costs_denom, axis=0)

        mult = np.exp((is_inside * (costs_inside - costs_denom_inside)) + \
                      (is_outside * (costs_outside - costs_denom_outside)))
        weights *= mult
        return weights / np.sum(weights)
    def reweight(self, theta, feasible):
        weights = np.array(self.weights)
        mult = np.zeros(self.NUM_PARTICLES)

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
        distances, indices = nbrs.kneighbors(feasible)
        max_dist = min(np.amax(distances), 0.5)
        distances, indices = nbrs.kneighbors(self.particles)
        # alpha = self.ALPHA_I
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            if distances[i][0] >= max_dist:
                alpha = self.ALPHA_O
            else:
                alpha = self.ALPHA_I
            # if particle[0] < np.amin(feasible) or particle[0] > np.amax(feasible):
            #     alpha = self.ALPHA_O
            # else:
            #     alpha = self.ALPHA_I
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
            new_particles.append(np.random.normal(sample, self.h))
            # new_particles.append(sample)
        self.particles = new_particles
        self.weights = [1/size]*size
        self.NUM_PARTICLES = size
    def distribution_mode(self):
        kernel = kde(np.array(self.particles).T)
        xs = []
        def kernel(x):
            DOF = len(x)
            cov = np.diag(np.ones(DOF)) * 0.5
            likelihoods = mvn.pdf(self.particles, mean=x, cov=cov)
            return np.sum(likelihoods) / self.NUM_PARTICLES
        cov = np.diag(np.ones(len(self.particles[0]))) * 0.2
        for _ in range(1):
            start = np.mean(self.particles, axis=0)
            # start += np.random.multivariate_normal(mean=start, cov=cov)
            # print "start: " + str(start)
            res = minimize(lambda x: -kernel(x), start)
            # print "end: " + str(res.x)
            xs.append(res.x)
        mode = max(xs, key=kernel)
        return mode
    def neg_log_likelihood_mean(self, data):
        # mode = self.distribution_mode()
        mode = np.mean(self.particles, axis=0)
        total = 0
        for (theta, feasible) in data:
            total += pe.prob_theta_given_lam_stable_set_weight_num(theta, mode, self.w, self.cost, 1)
            total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, mode, self.w, self.cost, 1)
        return -total
    def neg_log_likelihood(self, data):
        total = 0
        for (i, particle) in enumerate(self.particles):
            ll = 0
            for (theta, feasible) in data:
                ll += pe.prob_theta_given_lam_stable_set_weight_num(theta, particle, self.w, self.cost, 1)
                ll -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, particle, self.w, self.cost, 1)
            total += (ll * self.weights[i])
        return -total
    def entropy(self, num_boxes=10, axis_ranges=None):
        if axis_ranges is None:
            axis_ranges = np.array([3]*len(self.particles[0]))
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
        # left = np.amin(feasible)
        # right = np.amax(feasible)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
        distances, indices = nbrs.kneighbors(feasible)
        max_dist = min(np.amax(distances), 0.5)
        distances, indices = nbrs.kneighbors(self.particles)
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            if distances[i][0] >= max_dist:
                alpha = self.ALPHA_O
            else:
                alpha = self.ALPHA_I
            # if particle[0] < left or particle[0] > right:
            #     alpha = self.ALPHA_O
            # else:
            #     alpha = self.ALPHA_I
            theta = max(feasible, key=lambda x: pe.prob_theta_given_lam_stable_set_weight_num(x, particle, self.w, self.cost, alpha))
            weights = self.reweight(theta, feasible)
            d = SetWeightsParticleDistribution(self.particles, weights, self.cost, self.w, self.ALPHA_I, self.ALPHA_O)
            ent = d.entropy(num_boxes, axis_ranges)
            avg_ent += weight * ent
            # if i % 50 == 0:
            #     print "\r%d" % i,
            #     sys.stdout.flush()
        ret = self.entropy(num_boxes, axis_ranges) - avg_ent
        # print str(ret) + ": (" + str(np.amin(feasible)) + ", " + str(np.amax(feasible)) + ")"
        return ret
    def expected_cost(self, feasible):
        new_weights = np.zeros(self.NUM_PARTICLES)
        avg_cost = 0
        alpha = self.ALPHA_I
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            theta = max(feasible, key=lambda x: pe.prob_theta_given_lam_stable_set_weight_num(x, particle, self.w, self.cost, alpha))
            avg_cost += weight * self.cost(theta, particle, self.w)
        # print str(avg_cost) + ": (" + str(np.amin(feasible)) + ", " + str(np.amax(feasible)) + ")"
        return avg_cost
    def expected_cost2(self, feasible, mode):
        probs = pe.prob_theta_given_lam_stable_set_weight_num(feasible, mode, self.w, self.cost, self.ALPHA_I)
        probs -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, mode, self.w, self.cost, self.ALPHA_I)
        probs = np.exp(probs)
        costs = self.cost(feasible, mode, self.w)
        return np.sum(probs * costs)
    def info_gain_kl(self, feasible):
        avg = 0
        alpha = self.ALPHA_I
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            theta = max(feasible, key=lambda x: pe.prob_theta_given_lam_stable_set_weight_num(x, particle, self.w, self.cost, alpha))
            weights = self.reweight(theta, feasible)
            div = np.sum(weights * np.log(weights)) - np.log(1 / self.NUM_PARTICLES)
            avg += weight * div
        print str(avg) + ": (" + str(np.amin(feasible)) + ", " + str(np.amax(feasible)) + ")"
        return avg

class SetMeanParticleDistribution():
    def __init__(self, particles, weights, cost, m, ALPHA_I, ALPHA_O, h=None):
        self.NUM_PARTICLES = len(particles)
        self.particles = particles
        self.weights = weights
        self.cost = cost
        self.m = m
        self.ALPHA_I = ALPHA_I
        self.ALPHA_O = ALPHA_O
        if h is None:
            self.h = H
        else:
            self.h = h
    def reweight(self, theta, feasible):
        weights = np.array(self.weights)
        mult = np.zeros(self.NUM_PARTICLES)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
        distances, indices = nbrs.kneighbors(feasible)
        max_dist = min(np.amax(distances), 0.5)
        distances, indices = nbrs.kneighbors(self.particles)
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            if distances[i][0] >= max_dist:
                alpha = self.ALPHA_O
            else:
                alpha = self.ALPHA_I
            mult[i] = pe.prob_theta_given_lam_stable_set_weight_num(theta, self.m, particle, self.cost, alpha)
            mult[i] -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, self.m, particle, self.cost, alpha)
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
            new_particles.append(np.random.normal(sample, self.h))
            # new_particles.append(sample)
        self.particles = new_particles
        self.weights = [1/size]*size
        self.NUM_PARTICLES = size
        self.particles = self.particles / np.linalg.norm(self.particles, axis=1).reshape(-1, 1)
    def distribution_mode(self):
        kernel = kde(np.array(self.particles).T)
        xs = []
        def kernel(x):
            DOF = len(x)
            cov = np.diag(np.ones(DOF)) * 0.0625
            likelihoods = mvn.pdf(self.particles, mean=x, cov=cov)
            return np.sum(likelihoods) / self.NUM_PARTICLES
        for _ in range(7):
            start = np.random.uniform(size=len(self.particles[0]))
            start = start / np.linalg.norm(start)
            res = minimize(lambda x: -kernel(x), start)
            xs.append(res.x)
        mode = max(xs, key=kernel)
        return mode
    def neg_log_likelihood(self, data):
        total = 0
        for (i, particle) in enumerate(self.particles):
            ll = 0
            for (theta, feasible) in data:
                ll += pe.prob_theta_given_lam_stable_set_weight_num(theta, self.m, particle, self.cost, 1)
                ll -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, self.m, particle, self.cost, 1)
            total += (ll * self.weights[i])
        return -total
    def neg_log_likelihood_mean(self, data, i=0, title=''):
        # mean = np.mean(self.particles, axis=0)
        mode = self.distribution_mode()
        # mode = np.mean(self.particles, axis=0)
        # mode = mode / np.linalg.norm(mode)
        # if i == 9:
        #     fig, axes = plt.subplots(nrows=2, ncols=2)
        #     axes = np.ndarray.flatten(np.array(axes))
        #     for i, ax in enumerate(axes):
        #         theta, feasible = data[i]
        #         ax.set_yticklabels([])
        #         ax.set_xticklabels([])
        #         xx, yy = np.mgrid[-5:5:100j, -5:5:100j]
        #         positions = np.vstack([xx.ravel(), yy.ravel()])
        #         def likelihood(idx, point):
        #             alpha = self.ALPHA_I
        #             return pe.prob_theta_given_lam_stable_set_weight_num(point, self.m, mode, self.cost, alpha)\
        #             -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, self.m, mode, self.cost, alpha)
        #         likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
        #         # print np.amin(likelihoods)
        #         # print "max: " + str(np.amax(likelihoods))
        #         f = np.reshape(likelihoods.T, xx.shape)
        #         ax.imshow(np.flip(f, 1).T, cmap='inferno', interpolation='nearest', extent=(-5, 5, -5, 5), vmin=-33, vmax=0)
        #         ax.scatter(feasible[:,0], feasible[:,1], c='C0')
        #         ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        #         fig.suptitle(title)
        #         plt.pause(0.01)
        #         print pe.prob_theta_given_lam_stable_set_weight_num(theta, self.m, mode, self.cost, 1)\
        #         -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, self.m, mode, self.cost, 1)
        total = 0
        for (theta, feasible) in data:
            total += pe.prob_theta_given_lam_stable_set_weight_num(theta, self.m, mode, self.cost, 1)
            total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, self.m, mode, self.cost, 1)
        return -total
    def entropy(self, num_boxes=10, axis_ranges=None):
        if axis_ranges is None:
            axis_ranges = np.array([3]*len(self.particles[0]))
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
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
        distances, indices = nbrs.kneighbors(feasible)
        max_dist = min(np.amax(distances), 0.5)
        distances, indices = nbrs.kneighbors(self.particles)
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            if distances[i][0] >= max_dist:
                alpha = self.ALPHA_O
            else:
                alpha = self.ALPHA_I
            theta = max(feasible, key=lambda x: pe.prob_theta_given_lam_stable_set_weight_num(x, self.m, particle, self.cost, alpha))
            weights = self.reweight(theta, feasible)
            d = SetWeightsParticleDistribution(self.particles, weights, self.cost, self.m, self.ALPHA_I, self.ALPHA_O)
            ent = d.entropy(num_boxes, axis_ranges)
            avg_ent += weight * ent
            # if i % 50 == 0:
            #     print "\r%d" % i,
            #     sys.stdout.flush()
        ret = self.entropy(num_boxes, axis_ranges) - avg_ent
        # print str(ret) + ": (" + str(np.amin(feasible)) + ", " + str(np.amax(feasible)) + ")"
        return ret
    def expected_cost(self, feasible):
        new_weights = np.zeros(self.NUM_PARTICLES)
        avg_cost = 0
        alpha = self.ALPHA_I
        for i in range(self.NUM_PARTICLES):
            particle = self.particles[i]
            weight = self.weights[i]
            theta = max(feasible, key=lambda x: pe.prob_theta_given_lam_stable_set_weight_num(x, self.m, particle, self.cost, alpha))
            avg_cost += weight * self.cost(theta, self.m, particle)
        # print str(avg_cost) + ": (" + str(np.amin(feasible)) + ", " + str(np.amax(feasible)) + ")"
        return avg_cost
    def expected_cost2(self, feasible, mode):
        probs = pe.prob_theta_given_lam_stable_set_weight_num(feasible, self.m, mode, self.cost, self.ALPHA_I)
        probs -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, self.m, mode, self.cost, self.ALPHA_I)
        probs = np.exp(probs)
        costs = self.cost(feasible, self.m, mode)
        return np.sum(probs * costs)