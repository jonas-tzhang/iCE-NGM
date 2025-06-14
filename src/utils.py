import numpy as np
from tqdm import tqdm


def parabolic_1(X):
    b, e, kappa = 5, 0.1, 0.5
    g_X = b - X[:, 1] - kappa * (X[:, 0] - e) ** 2
    return g_X


def parabolic_2(X):
    b, e, kappa = 5, 0, 0.1
    g_X = b - X[:, 1] - kappa * (X[:, 0] - e) ** 2
    return g_X


def series_system(X):
    x1, x2 = X[:, 0], X[:, 1]
    val = np.zeros((X.shape[0], 4))
    val[:, 0] = 3 + (x1 - x2) ** 2 / 10 - (x1 + x2) / np.sqrt(2)
    val[:, 1] = 3 + (x1 - x2) ** 2 / 10 + (x1 + x2) / np.sqrt(2)
    val[:, 2] = x1 - x2 + 7 / np.sqrt(2)
    val[:, 3] = x2 - x1 + 7 / np.sqrt(2)

    return np.min(val, axis=1)


def high_dimensional(X):
    beta = 3.5

    n = X.shape[1]
    term_2 = 1/np.sqrt(n) * np.sum(X, axis=1)

    return np.minimum(beta - term_2, beta + term_2)


def repeat_experiments(ceais, nrep=100, Nf=3000, model_name="Unknown"):
    print("-" * 50)
    print("Experiment: {}, nrep={}, Nf={}".format(model_name, nrep, Nf))
    pf_list = []
    for _ in tqdm(range(nrep)):
        pf = ceais.predict(N=Nf)
        pf_list.append(pf)
    # Print the results.
    print("Mean Failure Probability pf = {:4e}".format(np.mean(pf_list)))
    print("Standard Deviation of pf = {:4e}".format(np.std(pf_list)))
    print("C.O.V. of pf = {:4e}".format(np.std(pf_list)/np.mean(pf_list)))
