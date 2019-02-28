import numpy as np


def gaussian_pdf(X, mean, std):
    '''
    Calculate the probability density of a
    point in a Gaussian, given a mean and standard deviation.

    Arguments:
        X: the input data
        mean: the mean of the gaussian
        std: the standard deviation of the gaussian
    '''
    pdfs = []
    for i in range(mean.shape[0]):
        pdf = []
        for j in range(X.shape[0]):
            det_sigma = np.linalg.det(2 * np.pi * std[i])
            coeff = (1. / det_sigma**0.5)

            inv_std = np.linalg.inv(std[i])
            x_minus_mean = X[j] - mean[i]

            power = -0.5 * \
                np.dot(np.dot(x_minus_mean, inv_std), x_minus_mean.T)

            prob = coeff * np.exp(power)
            pdf.append(prob)

        pdfs.append(pdf)
    pdfs = np.array(pdfs)
    return pdfs


class GaussianMixture(object):
    '''
    A class that represents a gaussian
    mixture model.
    '''

    def __init__(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        self.mu = None
        self.sigma = None
        self.phi = None

    def initialise(self, X):
        '''
        Given some data, this method initialises
        the parameters of the gaussian mixture model,
        i.e. the means, standard deviations, and mixture
        coefficients.

        Arguments:
            X: the input data.
        '''
        number_of_points = X.shape[0]
        number_of_dimensions = X.shape[1]

        indices = np.random.randint(
            number_of_points, size=self.number_of_clusters)

        self.mu = X[indices]
        self.sigma = np.array([np.identity(number_of_dimensions)
                               for i in range(self.number_of_clusters)])
        self.phi = np.ones((self.number_of_clusters, 1)) / \
            self.number_of_clusters

    def expectation_step(self, X):
        '''
        Performs the expectation step of the EM algorithm.
        i.e. calculates the posterior of the latents P(z|X)
        to find the responsibilities.

        Arguments:
                X: the input data

        Returns:
                responsibilities: the responsibilities, which
                describes the probabilities of each cluster for
                each given data point.
        '''
        r_probs = np.multiply(self.phi, gaussian_pdf(X, self.mu, self.sigma))
        responsibilities = np.divide(r_probs, np.sum(r_probs, axis=0))

        return responsibilities

    def maximisation_step(self, X, responsibilities):
        '''
        Performs the maximisation step of the EM algorithm.
        i.e. finds the parameters that maximises the likelihood
        of the data.

        Arguments:
                X: the input data
                responsibilities: the responsibilities
        '''
        responsibilities_sum = np.sum(responsibilities, axis=1).reshape(-1, 1)
        self.phi = responsibilities_sum / X.shape[0]
        self.mu = np.divide(np.dot(responsibilities, X), responsibilities_sum)

        new_sigma = np.zeros((self.number_of_clusters, X.shape[1], X.shape[1]))

        for i in range(X.shape[0]):
            for j in range(self.number_of_clusters):

                cov_ij = responsibilities[j][
                    i] * np.outer((X[i] - self.mu[j]), (X[i] - self.mu[j]).T)

                new_sigma[j] += cov_ij / np.sum(responsibilities[j])

        self.sigma = new_sigma

    def fit(self, X, threshold=1E-10):
        '''
        Performs the EM algorithm to find the parameters
        that maximise the likelihood of the data.

        Arguments:
            X: the input data
            iters: the number of iterations to perform EM.

        Returns:
           mu: the means of the clusters
           sigma: the std of the clusters
           phi: the mixture coefficients
        '''
        self.initialise(X)

        # Initialise the change in mean
        change_in_mu = 1E10

        # While the change in mu is more than the threshold
        while change_in_mu > threshold:
            old_mu = self.mu

            # Find the probability for each cluster, given the data points
            responsibilities = self.expectation_step(X)

            # Find parameters that maximise the new assignments
            self.maximisation_step(X, responsibilities)

            new_mu = self.mu

            change_in_mu = np.linalg.norm(new_mu - old_mu)

        mu = self.mu
        sigma = self.sigma
        phi = self.phi

        return mu, sigma, phi
