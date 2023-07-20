import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
import scipy.special
from scipy.stats import binom
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from multiprocessing import Pool

from itertools import repeat
import functools


def eta_partial(kappa):
    """
    Generate Bernoulli(1/2) on marked kappa star
    """
    multiplicity = np.fromfunction(
        lambda _, k: scipy.special.comb(kappa, k), (2, kappa + 1)
    )
    eta = multiplicity / (2 ** (kappa + 1))
    return eta


def hamiltonian_partial(beta, B, l):
    """
    Hamiltonian matrix for marked l star
    """
    return np.fromfunction(
        lambda x, j: (2 * x - 1) * (beta / 2 * (2 * j - l) + B), shape=(2, l + 1)
    )


def get_marginal(name, p):
    """
    Get the marginal distribution in a tree
    """
    assert name in {"a", "v"}
    return np.sum(p, axis=(2, 3) if name == "a" else (0, 1))


def edge_distribution_partial(p, l):
    """
    Calculate edge distribution \pi_p
    """
    counts = np.vstack([l - np.arange(l + 1), np.arange(l + 1)])
    return p @ counts.T / l


def edge_distribution_partial_linear(l):
    """
    Linear form for edge distribution
    """

    def get_counts(x0, xv, x, j):
        return np.where(x0 == x, l - j + xv * (2 * j - l), 0)

    return np.fromfunction(get_counts, (2, 2, 2, l + 1)) / l


def admissibility_linear(l, k):
    """
    Linear constraint for admissibility given a flattened \mu^a and \mu^v
    """
    linear_a = edge_distribution_partial_linear(l).reshape(2, 2, 2 * (l + 1))
    linear_v = edge_distribution_partial_linear(k).reshape(2, 2, 2 * (k + 1))
    return np.block(
        [
            [linear_a[0, 0], -linear_v[0, 0]],
            [linear_a[0, 1], -linear_v[1, 0]],
            [linear_a[1, 0], -linear_v[0, 1]],
            [linear_a[1, 1], -linear_v[1, 1]],
        ]
    )


def edge_distribution_partial_test(p, l):
    """
    Test that edge_distribution_partial_linear produces the same edge distribution
    """
    pi_mu = edge_distribution_partial(p, l)
    counts = edge_distribution_partial_linear(l)
    pi_mu2 = np.sum(counts * p, axis=(2, 3))

    assert np.all(np.abs(pi_mu - pi_mu2) < 1e-10)


def check_distibution_partial(pi_mu, l):
    pi_mu_1 = np.sum(pi_mu, axis=0)
    leaf_dist = binom.pmf(np.arange(l + 1), l, pi_mu_1[1])
    return np.repeat([leaf_dist], 2, axis=0) / 2


def hat_distibution_partial(pi_mu, l):
    pi_mu_0 = np.sum(pi_mu, axis=1)

    leaf_dist_neg = binom.pmf(np.arange(l + 1), l, pi_mu[0, 1] / (pi_mu_0[0] + 1e-15))
    leaf_dist_pos = binom.pmf(np.arange(l + 1), l, pi_mu[1, 1] / (pi_mu_0[1] + 1e-15))
    return np.vstack([leaf_dist_neg, leaf_dist_pos]) / 2


def relative_entropy(p, q):
    p_divided_by_q = np.divide(p, q, out=np.ones_like(p), where=(p > 0) & (q > 0))
    partial = p * np.log(p_divided_by_q)
    return np.sum(partial)


def objective_function_partial(mu, beta, B, l, _hamiltonian=None):
    """
    Objective function for one of the marked stars mu
    """
    if _hamiltonian is None:
        _hamiltonian = hamiltonian_partial(beta, B, l)
    _hamiltonian = np.sum(_hamiltonian * mu)

    pi_mu = edge_distribution_partial(mu, l)
    ## TEMPORARY TEST ##
    # edge_distribution_partial_test(mu, l)

    check_dist = check_distibution_partial(pi_mu, l)
    hat_dist = hat_distibution_partial(pi_mu, l)

    return (
        _hamiltonian
        - (relative_entropy(mu, check_dist) + relative_entropy(mu, hat_dist)) / 2
    )


def objective_function(
    mu_a, mu_v, beta, B, l, k, hamiltonian_a=None, hamiltonian_v=None
):
    """
    Full objective function for both marked stars a and v
    """
    if hamiltonian_a is None:
        hamiltonian_a = hamiltonian_partial(beta, B, l)
    if hamiltonian_v is None:
        hamiltonian_v = hamiltonian_partial(beta, B, k)
    alpha = k / l
    mu_a_obj = objective_function_partial(mu_a, beta, B, l, hamiltonian_a)
    mu_v_obj = objective_function_partial(mu_v, beta, B, k, hamiltonian_v)
    return alpha * mu_a_obj + mu_v_obj


def flat_to_2d(mu, l, k):
    return mu[: 2 * (l + 1)].reshape(2, l + 1), mu[2 * (l + 1) :].reshape(2, k + 1)


def visualize(p, ax, vmin=None, vmax=None):
    """
    Generates a heatmap for the distribution
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(
        p, extent=[0, 2 * p.shape[1], 0, 4], cmap="Greys", vmin=vmin, vmax=vmax
    )
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            text = ax.text(
                2 * j + 1,
                4 - (2 * i + 1),
                f"{p[i, j]:.3f}",
                ha="center",
                va="center",
                color="red",
            )
    return im


@functools.cache
def get_multiplicities(kappa):
    # Get the equivalent configuration for each combined vertex-edge state
    def counts(*vertex_marks):
        stacked = np.stack(vertex_marks, axis=0)
        return np.sum(stacked, axis=0)

    ks = np.fromfunction(counts, [2] * kappa, dtype=np.int64)

    return ks


def vertex_distribution(p, kappa):
    """
    Joint distribution for all edge-vertex pairs

    Input:
    p: 1-d array representing distribution on leaf configurations

    Output:
    kappa-d array representing the joint distribution of all edge-vertex pairs
    """
    # Match the configurations with the ones from configurations list
    ks = get_multiplicities(kappa)
    k_reshaped = ks.flatten()

    # Calculate probabilities for each one
    probabilities = [p[k] / scipy.special.comb(kappa, k) for k in k_reshaped]

    return np.array(probabilities).reshape((2,) * kappa)


def optimize(
    beta,
    B,
    l,
    k,
    mu0=None,
    verbose=False,
    ftol=1e-9,
    max_iter=1000,
    project_mu0=False,
    constraint_tol=0,
    num_trials=1,
):
    hamiltonian_a = hamiltonian_partial(beta, B, l)
    hamiltonian_v = hamiltonian_partial(beta, B, k)

    bounds = Bounds(np.zeros(2 * (l + k + 2)), np.ones(2 * (l + k + 2)))
    norm_constraint = LinearConstraint(
        np.block(
            [
                [np.ones(2 * (l + 1)), np.zeros(2 * (k + 1))],
                [np.zeros(2 * (l + 1)), np.ones(2 * (k + 1))],
            ]
        ),
        1 - constraint_tol,
        1 + constraint_tol,
    )
    admissibility_constraint = LinearConstraint(
        admissibility_linear(l, k), lb=-constraint_tol, ub=constraint_tol
    )

    if verbose:
        print(f"{beta:.3f=}, {B:.3f=}, {l=}, {k=}")

    max_objective = -np.inf

    for trial in range(num_trials):
        success = False

        while not success:
            mu0 = np.random.uniform(-1, 1, size=2 * (l + k + 2))
            mu0[: 2 * (l + 1)] = mu0[: 2 * (l + 1)] / np.sum(mu0[: 2 * (l + 1)])
            mu0[2 * (l + 1) :] = mu0[2 * (l + 1) :] / np.sum(mu0[2 * (l + 1) :])
            # Project onto the closest point on the probability simplex
            if project_mu0:
                res = minimize(
                    lambda mu: np.sum((mu - mu0.flatten()) ** 2) / 2,
                    x0=mu0.flatten(),
                    method="trust-constr",
                    constraints=[norm_constraint, admissibility_constraint],
                    bounds=bounds,
                )
                # mu0 = res.x.reshape(2, -1)
                if not res.success:
                    print(f"Projection unsuccessfull: {res.message}, trying again")
                    print(f"{beta=:.3f}, {B=:.3f}, {l=}, {k=}")

            if verbose:
                mu0_a, mu0_v = flat_to_2d(mu0, l, k)
                print(f"{mu0_a=}\n{mu0_v=}")

            res = minimize(
                lambda mu: -objective_function(
                    *flat_to_2d(mu, l, k), beta, B, l, k, hamiltonian_a, hamiltonian_v
                ),
                x0=mu0.flatten(),
                method="SLSQP",
                constraints=[norm_constraint, admissibility_constraint],
                bounds=bounds,
                options={"ftol": ftol, "disp": verbose, "maxiter": max_iter},
            )

            if not res.success:
                # print(f"Main optimization unsuccessfull: {res.message}, trying again")
                # print(f"{beta=:.3f}, {B=:.3f}, {l=}, {k=}")
                pass
            else:
                success = True

        mu_a, mu_v = flat_to_2d(res.x, l, k)
        objective = objective_function(mu_a, mu_v, beta, B, l, k)

    if objective > max_objective:
        max_objective = objective
        max_mu_a, max_mu_v = mu_a, mu_v

    if verbose:
        print(f"{max_objective=}\n")
        print(
            f"{max_mu_a=}\n{max_mu_v=}\n\n{np.sum(max_mu_a)=}\n{np.sum(max_mu_v)=}\n\n{edge_distribution_partial(max_mu_a, l)=}\n{edge_distribution_partial(max_mu_v, k)=}"
        )

    return max_mu_a, max_mu_v


def vertex_mutual_info(mu, l):
    ver_dist = np.sum(vertex_distribution(mu, l), axis=tuple(i for i in range(2, l)))
    ver_dist_x1 = np.sum(ver_dist, axis=1)
    ver_dist_x2 = np.sum(ver_dist, axis=0)
    return relative_entropy(ver_dist, np.outer(ver_dist_x1, ver_dist_x2))


def calc_mutual_info(args):
    beta, B, l, k = args
    mu_a, mu_v = optimize(
        beta, B, l, k, ftol=1e-10, constraint_tol=1e-15, num_trials=10
    )

    mu0a = np.sum(mu_a, axis=1)
    mu0v = np.sum(mu_v, axis=1)
    return (
        mu_a,
        mu_v,
        vertex_mutual_info(mu_a[0] / (mu0a[0] + 1e-15), l),
        vertex_mutual_info(mu_a[1] / (mu0a[1] + 1e-15), l),
        vertex_mutual_info(mu_v[0] / (mu0v[0] + 1e-15), k),
        vertex_mutual_info(mu_v[1] / (mu0v[1] + 1e-15), k),
    )


def job(l, k, seed):
    np.random.seed(seed)

    betas = np.linspace(-1, 1, 31)
    Bs = np.linspace(-1, 1, 31)
    betam, Bm = np.meshgrid(betas, Bs)

    # with Pool(processes=4) as pool:
    #     mutual_info_a_0 = pool.map(
    #         func, tqdm(zip(betam.flatten(), Bm.flatten()), total=len(betam.flatten()))
    #     )
    res = process_map(
        calc_mutual_info,
        zip(betam.flatten(), Bm.flatten(), repeat(l), repeat(k)),
        total=len(betam.flatten()),
    )
    mu_a, mu_v, mia0, mia1, miv0, miv1 = zip(*res)
    with open(f"../data/ising_bptt/{l=}_{k=}.npz", "wb+") as f:
        np.savez(
            f,
            betas=betas,
            Bs=Bs,
            l=l,
            k=k,
            betam=betam,
            Bm=Bm,
            seed=seed,
            mu_a=mu_a,
            mu_v=mu_v,
            mia0=mia0,
            mia1=mia1,
            miv0=miv0,
            miv1=miv1,
        )


if __name__ == "__main__":
    for l in range(4, 7):
        for k in range(l, 7):
            print(f"{l=}, {k=}")
            job(l=l, k=k, seed=0)
