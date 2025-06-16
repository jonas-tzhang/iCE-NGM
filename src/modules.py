import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import linalg


class GaussianMixtureDensity:
    def __init__(self,
                 p=1, K=1,
                 weights_init=None, mean_init=None, covariance_init=None,
                 eigenvalue_bound=None, tol=3e-1, max_iter=100):
        """
        Gaussian Mixture Density class with the following feature:
        - can be trained with constrained monotone EM algorithm.
        - is designed mainly for importance sampling, but can also be used to process weighted data.
        """

        """ 1. Setting the hyperparameters of the GMM. """
        self.p, self.K = p, K   # dimensionality and number of components
        self.eigenvalue_bound = eigenvalue_bound if eigenvalue_bound is not None else [0, np.inf]   # eigenvalue bound
        self.tol, self.max_iter = tol, max_iter   # tolerance and maximum number of iterations

        """ 2. Setting the initial values of parameters to be optimized. """
        self.weights = weights_init.copy()/np.sum(weights_init) if weights_init is not None else np.ones(K) / K   # weights
        self.mean = mean_init.copy() if mean_init is not None else np.zeros((K, p))
        self.cov = covariance_init.copy() if covariance_init is not None \
            else np.array([np.eye(p) for _ in range(K)])

    def fit(self, X, data_weights=None, gmm_true=None):
        """ Fit the GMM to the data X. """

        """ 1. Initialize parameters. """
        tol, max_iter = self.tol, self.max_iter
        data_weights_ = np.ones(X.shape[0]) if data_weights is None else data_weights
        weighted_llh = [self.weighted_llh(X, data_weights_)]

        """ 2. Iterate until convergence. """
        for iter_num in range(max_iter):
            """ 2.1. E-step: calculate the responsibilities. """
            resp = self.responsibility(X)

            """ 2.2. M-step: update the parameters. """
            # update the weights
            self.weights = np.average(resp, axis=0, weights=data_weights_)
            wts = resp * data_weights_[:, np.newaxis]
            # update the mean
            weights_sum = np.sum(wts, axis=0)
            weighted_sum = np.dot(wts.T, X)
            self.mean = weighted_sum / weights_sum[:, np.newaxis]

            # update the covariance
            for k in range(self.K):
                try:
                    self.cov[k] = (X-self.mean[k]).T @ np.diag(wts[:, k]/weights_sum[k]) @ (X-self.mean[k])
                except RuntimeWarning:
                    print("RuntimeWarning: covariance matrix is not positive definite.")

                self.cov[k] = self.eigenvalue_threshold(self.cov[k])

            """ 2.3. Check convergence. """
            weighted_llh.append(self.weighted_llh(X, data_weights_))
            # print("Iteration {}: weighted log-likelihood = {}".format(iter_num+1, weighted_llh[-1]))
            if np.abs(weighted_llh[-1] - weighted_llh[-2])/np.abs(weighted_llh[-2]) < 0.1*tol\
                    or np.abs(weighted_llh[-1] - weighted_llh[-2]) < tol:
                # print("Converged after {} iterations.".format(iter_num+1))
                break
            elif iter_num == max_iter-1:
                # print("Maximum number of iterations reached.")
                pass

    def eigenvalue_threshold(self, cov):
        """ Threshold the eigenvalues of the covariance matrix.
        """
        # calculate the eigenvalues and eigenvectors using scipy.linalg.eigh
        eig_val, eig_vec = linalg.eigh(cov)
        # threshold the eigenvalues
        eig_val = np.clip(eig_val, self.eigenvalue_bound[0], self.eigenvalue_bound[1])
        # return the thresholded covariance matrix
        return eig_vec @ np.diag(eig_val) @ eig_vec.T

    def weighted_llh(self, X, data_weights=None):
        """ Calculate the weighted log-likelihood of X.
        """
        data_weights = np.ones(X.shape[0]) if data_weights is None else data_weights
        return np.sum(np.log(self.pdf(X)) * data_weights)

    def pdf(self, X):
        """ Calculate the pdf of X.
        """
        pdf = np.zeros(X.shape[0])
        for k in range(self.K):
            pdf += self.weights[k] * multivariate_normal.pdf(X, mean=self.mean[k], cov=self.cov[k])
        return pdf

    def sample(self, N):
        """ Sample N points from the GMM.
        """
        # pre-allocation
        X = np.zeros((N, self.p))
        # sample the number of points for each component
        counts = np.random.multinomial(N, self.weights)

        cum_idx = 0
        for k in range(self.K):
            count = counts[k]
            if count > 0:
                X[cum_idx:cum_idx + count] = multivariate_normal.rvs(
                    mean=self.mean[k], cov=self.cov[k], size=count)
                cum_idx += count

        return X

    def responsibility(self, X):
        """ Calculate the responsibility of each component for each point in X.
        """
        resp = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            resp[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.mean[k], cov=self.cov[k])
        return resp / np.sum(resp, axis=1).reshape(-1, 1)

    @staticmethod
    def plot(X_input, gmm_true, gmm_est=None):
        # decide the plot range from the input data.
        x_min = np.min(X_input[:, 0]) - 2
        x_max = np.max(X_input[:, 0]) + 2
        y_min = np.min(X_input[:, 1]) - 2
        y_max = np.max(X_input[:, 1]) + 2
        # plot the sample and contour of the true density.
        plt.figure()
        plt.scatter(X_input[:, 0], X_input[:, 1], s=1)
        # contour plot of the self.is_density
        x = np.linspace(x_min, x_max, 1000)
        y = np.linspace(y_min, y_max, 1000)
        X, Y = np.meshgrid(x, y)
        Z_is_density = gmm_true.pdf(np.c_[X.ravel(), Y.ravel()])
        Z_is_density = Z_is_density.reshape(X.shape)
        plt.contour(X, Y, Z_is_density, colors='red')

        if gmm_est is not None:
            # plot the contour of the fitted GMM.
            Z_is_density_est = gmm_est.pdf(np.c_[X.ravel(), Y.ravel()])
            Z_is_density_est = Z_is_density_est.reshape(X.shape)
            plt.contour(X, Y, Z_is_density_est, colors='blue', linestyles='dashed')

        plt.show()

    def __str__(self):
        return "GaussianMixtureDensity(p={}, K={}, mean={}, cov={}, weights={})".format(
            self.p, self.K, self.mean, self.cov, self.weights)

    def __repr__(self):
        return self.__str__()