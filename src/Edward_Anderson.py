import numpy as np
from scipy.optimize import minimize

beta = 1
B = 0


def Hamiltonian(*state):
    x0, x1, x2, y01, y02 = 2 * np.array(state) - 1
    return beta / 2 * (x0 * x1 * y01 + x0 * x2 * y02) + B * x0


def target_function(x):
    mu = np.exp(x)
    exp = np.sum(np.fromfunction(Hamiltonian, (2, 2, 2, 2, 2)) * mu)
    rel_entr = np.sum(
        np.log(mu) * mu
    )  # Since eta is uniform, the KL-div is just the entropy

    mu_marg = np.sum(mu, axis=(0, 1, 2, 4))
    rel_marg_entr = np.sum(np.log(mu_marg) * mu_marg)

    return exp - rel_entr + rel_marg_entr


def grad_target_function(x):
    mu = np.exp(x)
    mu_marg = np.sum(mu, axis=(0, 1, 2, 4))
    grad_exp = np.fromfunction(Hamiltonian, (2, 2, 2, 2, 2))
    grad_rel_entr = np.log(mu) + 1
    grad_rel_marg_entr = np.log(mu_marg) + 1
    # repeat the values to all dimensions
    grad_rel_marg_entr = np.tile(
        np.tile(grad_rel_marg_entr[:, np.newaxis], (1, 2)), (2, 2, 2, 1, 1)
    )
    return (grad_exp - grad_rel_entr + grad_rel_marg_entr) * mu


def constraint_function(x):
    return np.sum(np.exp(x)) - 1


x = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
lam = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
patience = 1000
eps = 1e-4
target = -np.inf
for it in range(patience):
    print(f"{it=}")

    proximal_target = -np.inf

    lambda x: -target_function(x.reshape((2, 2, 2, 2, 2))) - lam * constraint_function(
        x.reshape((2, 2, 2, 2, 2))
    ) + constraint_function(x.reshape((2, 2, 2, 2, 2))) ** 2 / 2
    res = minimize(proximal_target_function, x0=x)
    x = res.x.reshape((2, 2, 2, 2, 2))

    # for _ in range(100):
    #     # Calculate the value
    #     new_proximal_target = target_function(x) + np.sum(
    #         lam * np.exp(x) - constraint_function(x) ** 2 / 2
    #     )
    #     print(f"{new_proximal_target=}")

    #     # Determine point of exit
    #     # if np.abs(proximal_target - new_proximal_target) < eps:
    #     #     break
    #     proximal_target = new_proximal_target

    #     # Calculate Gradient
    #     grad_proximal_target = grad_target_function(x) + lam * np.exp(x) - np.exp(x)
    #     x = x + 1e-4 * grad_proximal_target

    new_target = target_function(x)
    print(f"{new_target=}")
    if np.abs(target - new_target) < eps:
        break
    target = new_target
    lam = lam - constraint_function(x)

print(np.exp(x))
