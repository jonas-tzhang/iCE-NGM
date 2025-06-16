import numpy as np

from src.modules import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
from abc import ABCMeta, abstractmethod
from scipy.stats import norm
import copy


class ImprovedCEAIS(metaclass=ABCMeta):
    def __init__(self, limit_state_func, input_density):
        # variables defining the problem.
        self.limit_state_func = limit_state_func
        self.input_density = input_density
        self.sample_budget = None           # sample budget for the budget-based stopping criteria.
        self.sample_budget_leftover = None  # leftover sample budget for the budget-based stopping criteria.

        # variables storing algorithm results.
        self.is_density = None
        self.optimal_is_density = None
        self.X = None       # samples for the last iteration.
        self.g_X = None     # limit state function values for the last iteration.
        self.num_steps = 0  # number of iterations.
        self.final_sigma = None

        # variables for convergence check.
        self.delta = [np.inf]
        self.status = 0
        self.est_pred_cv = [np.inf]  # for budget-based stopping criteria.

    def fit(self, max_iter=100, samples_per_level=1000, delta_target=1.5, plot_process=False):
        """ This function fits the importance sampling density with fixed stopping criteria. """

        """ 1. Initialization. """
        self.initialize_is_density()
        sigma = 100

        """ 2. Iteration. """
        for i in range(max_iter):
            """ 2.1 Sampling. """
            # sampling.
            X = self.is_density.sample(samples_per_level)
            g_X = self.limit_state_func(X)
            # plot
            if plot_process:
                self.plot(X_input=X, title="Iteration {}, sigma = {}".format(i+1, sigma))

            """ 2.2 Check for convergence. """
            self.check_convergence_fixed_criteria(g_X, sigma, delta_target=delta_target)
            # save information for the last iteration.
            self.optimal_is_density = self.is_density
            self.X = X
            self.g_X = g_X
            self.num_steps = i + 1
            self.final_sigma = sigma

            # print("Sigma: ", sigma)

            # check convergence.
            if self.status == 1:
                print("Converged after {} iterations.".format(i+1))
                break

            """ 2.4 Update sigma. """
            sigma = self.find_sigma(X, g_X, sigma)

            """ 2.5 Update the importance sampling density. """
            self.update_is_density(X, g_X, sigma)
        
        if self.status == 0:
            print("Not converged after {} iterations.".format(max_iter))

    def check_convergence_fixed_criteria(self, g_X, sigma, delta_target=1.5):
        """ This function implements the fixed stopping criteria. """


        # compute the weights for convergence check.
        converge_weights = np.zeros_like(g_X)
        mask = g_X <= 0
        converge_weights[mask] = 1 / norm.cdf(-g_X[mask] / sigma)
        # compute the CV of weights.
        if np.mean(converge_weights) == 0:
            delta = np.inf
        else:
            delta = np.std(converge_weights) / np.mean(converge_weights)  # CV of weights

        self.delta.append(delta)

        # print(delta)

        if delta <= delta_target:
            self.status = 1

    def fit_with_budget(self, max_iter=1000, samples_per_level=1000, sample_budget=10000, plot_process=False,
                        delta_target=1.5, delta_upper_bound=2):
        """ This function fits the importance sampling density with budget-based stopping criteria. """

        """ 1. Initialization. """
        self.initialize_is_density()
        self.sample_budget = sample_budget
        self.sample_budget_leftover = sample_budget
        sigma = 100

        """ 2. Iteration. """
        for i in range(max_iter):
            # check if the sample budget is exceeded.
            if self.sample_budget_leftover < samples_per_level:
                # print("The left sample budget is less than the sample size. ")
                break

            """ 2.1 Sampling. """
            # sampling.
            X = self.is_density.sample(samples_per_level)
            g_X = self.limit_state_func(X)
            self.sample_budget_leftover -= samples_per_level
            # plot
            if plot_process:
                self.plot(X_input=X, title="Iteration {}, sigma = {}".format(i+1, sigma))

            """ 2.2 Check for convergence. """
            self.check_convergence_budget_based_criteria(g_X, sigma,
                                                         self.sample_budget_leftover+samples_per_level,
                                                         delta_target=delta_target, delta_upper_bound=delta_upper_bound)
            # save information for the last iteration.
            self.optimal_is_density = self.is_density
            self.X = X
            self.g_X = g_X
            self.final_sigma = sigma
            self.num_steps = i + 1
            # check convergence.
            if self.status == 1:
                print("Converged after {} iterations.".format(i + 1))
                break

            """ 2.3 Update sigma. """
            sigma = self.find_sigma(X, g_X, sigma)

            """ 2.4 Update the importance sampling density. """
            self.update_is_density(X, g_X, sigma)

        if self.status == 0:
            print("Not converged after {} iterations.".format(max_iter))

    def check_convergence_budget_based_criteria(self, g_X, sigma, samples_for_pred,
                                                delta_target=0.6, delta_upper_bound=1):
        """ This function implements the budget constrained stopping criteria. """

        # compute the weights for convergence check.
        converge_weights = np.zeros_like(g_X)
        mask = g_X <= 0
        converge_weights[mask] = 1 / norm.cdf(-g_X[mask] / sigma)
        # compute the CV of weights.
        if np.mean(converge_weights) == 0:
            delta = np.inf
        else:
            delta = np.std(converge_weights) / np.mean(converge_weights)  # CV of weights

        self.delta.append(delta)
        est_pred_cv = delta / np.sqrt(samples_for_pred)
        self.est_pred_cv.append(est_pred_cv)

        # print(delta)

        # budget constrained stopping criteria.
        if delta > delta_upper_bound:
            return

        if delta <= delta_target:
            self.status = 1
            return

        if ~np.isinf(self.est_pred_cv[-1]) and ~np.isinf(self.est_pred_cv[-2]):
            est_pred_cv_normalized = np.array(self.est_pred_cv)
            est_pred_cv_normalized = est_pred_cv_normalized[np.isfinite(self.est_pred_cv)]
            est_pred_cv_normalized = est_pred_cv_normalized / est_pred_cv_normalized[0]

            # print(est_pred_cv_normalized)

            if est_pred_cv_normalized[-1] - est_pred_cv_normalized[-2] > - 0.30:
                self.status = 1
            return

    @abstractmethod
    def initialize_is_density(self):
        """ This function initializes the importance sampling density. """
        pass

    @abstractmethod
    def update_is_density(self, X, g_X, sigma):
        """ This function updates the importance sampling density. """
        pass

    def predict(self, N=1000):
        """ This function predicts the failure probability. """

        # count the number of samples needed.
        N_pre = self.X.shape[0]
        if N > N_pre:
            # sample the remaining points.
            X = self.optimal_is_density.sample(N - N_pre)
            g_X = self.limit_state_func(X)
            # combine.
            X = np.vstack((self.X, X))
            g_X = np.hstack((self.g_X, g_X))
        else:
            X = self.X[:N]
            g_X = self.g_X[:N]

        W_X = self.input_density.pdf(X) / self.optimal_is_density.pdf(X)

        pf = np.mean((g_X <= 0) * W_X)

        return pf

    def predict_with_budget(self):
        """ This function predicts the failure probability with the budget-based stopping criteria. """

        # sample the remaining points.
        X = self.optimal_is_density.sample(self.sample_budget_leftover)
        g_X = self.limit_state_func(X)
        # combine.
        X = np.vstack((self.X, X))
        g_X = np.hstack((self.g_X, g_X))

        W_X = self.input_density.pdf(X) / self.optimal_is_density.pdf(X)

        pf = np.mean((g_X <= 0) * W_X)

        return pf

    def find_sigma(self, X, g_X, sigma_old):
        """ This function finds the optimal sigma. """
        X_pdf_input = self.input_density.pdf(X)
        X_pdf_is = self.is_density.pdf(X)

        def objective(sigma_new):
            sample_weights = norm.cdf(-g_X / sigma_new) * X_pdf_input / X_pdf_is
            delta = np.std(sample_weights) / np.mean(sample_weights)

            return (delta - 1.5)**2

        result = minimize_scalar(objective, method='bounded', bounds=(0, sigma_old))

        return result.x

    def plot(self, X_input=None, title=None):
        """ This function plots the samples and the contour of the self.is_density. """

        """ 1. scatter plot of the samples. """
        if X_input is not None:
            plt.scatter(X_input[:, 0], X_input[:, 1], s=1)
        """ 2. contour plot of the self.is_density. """
        x = np.linspace(-8, 8, 1000)
        y = np.linspace(-8, 8, 1000)
        X, Y = np.meshgrid(x, y)
        Z_is_density = self.is_density.pdf(np.c_[X.ravel(), Y.ravel()])
        Z_is_density = Z_is_density.reshape(X.shape)
        plt.contour(X, Y, Z_is_density, colors='red')
        """ 3.Plot the limit state function boundary. """
        Z_state_func = self.limit_state_func(np.c_[X.ravel(), Y.ravel()])
        Z_state_func = Z_state_func.reshape(X.shape)
        plt.contour(X, Y, Z_state_func, levels=[0], colors='b', linestyles='dashed')

        if title is not None:
            plt.title(title)
        plt.show()

    def plot_convergence(self):
        """ This function plots the convergence information. """
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(len(self.delta)), self.delta, color='tab:blue', label='delta')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('delta', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(self.delta)), self.est_pred_cv, color='tab:red', label='cov_est')
        ax2.set_ylabel('cov_est', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        fig.tight_layout()
        # legend
        fig.legend()
        plt.show()


class ImprovedCEAISNonparametricGaussianMixture(ImprovedCEAIS):

    def __init__(self, limit_state_func, input_density):
        super().__init__(limit_state_func=limit_state_func, input_density=input_density)
        self.opt_eigenvalue_lb = 1

    def initialize_is_density(self):
        """ This function initializes the importance sampling density. """
        self.is_density = GaussianMixtureDensity(p=self.input_density.p, K=1)

        self.all_is_density = []
        self.all_is_density.append(copy.deepcopy(self.is_density))

    def update_is_density(self, X, g_X, sigma):
        """ This function updates the importance sampling density. """

        W = norm.cdf(-g_X / sigma) * self.input_density.pdf(X) / self.is_density.pdf(X)
        # remove data points with zero weights, this step is to prevent numerical issue without
        # affecting the results.
        X = X[W > 0]
        W = W[W > 0]
        # use cross validation to find the optimal eigenvalue lower bound.
        self.opt_eigenvalue_lb = self.eigenvalue_lb_optimization(X, W, self.opt_eigenvalue_lb)

        # construct a GM with failure samples.
        self.is_density = GaussianMixtureDensity(p=X.shape[1], K=X.shape[0],
                                                 weights_init=W, mean_init=X,
                                                 eigenvalue_bound=[self.opt_eigenvalue_lb, 10])
        self.is_density.fit(X, data_weights=W)

        self.all_is_density.append(copy.deepcopy(self.is_density))

    @staticmethod
    def eigenvalue_lb_optimization(X, W, lb_opt_upper_bound):
        """ This function selects the optimal eigenvalue lower bound for the Gaussian Mixture Density. """

        num_fold = 5
        kf = KFold(n_splits=num_fold)
        optimization_steps = []

        def objective(eigenval_lb):
            llh = 0
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                W_train, W_test = W[train_index], W[test_index]
                gmm = GaussianMixtureDensity(p=X_train.shape[1], K=X_train.shape[0],
                                             weights_init=W_train, mean_init=X_train,
                                             eigenvalue_bound=[eigenval_lb, 10])
                gmm.fit(X_train, data_weights=W_train)
                llh += gmm.weighted_llh(X_test, data_weights=W_test)

            optimization_steps.append((eigenval_lb, -llh / num_fold))
            # print(f"Trial {len(optimization_steps)}, "
            #       f"eigenvalue Lower Bound: {eigenval_lb}, Average Log-Likelihood: {llh / num_fold}")

            return -llh / num_fold  # use negative log-likelihood

        # execute the golden section search
        result = minimize_scalar(objective,
                                 method='bounded',
                                 bounds=(0.01, lb_opt_upper_bound),
                                 options={'xatol': 2e-3, 'maxiter': 3})
        # print("The optimal eigenvalue lower bound is {}".format(result.x))

        return result.x









