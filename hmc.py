"""
Lab 2: Hamiltonian Monte Carlo Implementation

This file contains the functions you need to implement for the HMC lab.
Fill in the TODO sections and test your implementation in the notebook.

Hint: If you're unsure about JAX patterns (random keys, jnp operations, etc.),
look at the random walk implementation in the notebook for reference!
"""

import jax.numpy as jnp
from jax import grad, random


def leapfrog(theta, rho, log_prob_fn, epsilon, L):
    """Run L steps of leapfrog integration.

    The leapfrog integrator simulates Hamiltonian dynamics, which conserves
    energy (up to discretization error). Each step consists of:
        1. Half step for momentum: rho = rho + (epsilon/2) * grad_log_prob(theta)
        2. Full step for position: theta = theta + epsilon * rho
        3. Half step for momentum: rho = rho + (epsilon/2) * grad_log_prob(theta)

    Args:
        theta: Initial position, shape (D,)
        rho: Initial momentum, shape (D,)
        log_prob_fn: Function that returns log probability
        epsilon: Step size
        L: Number of leapfrog steps

    Returns:
        theta_new: Final position, shape (D,)
        rho_new: Final momentum, shape (D,)
    """
    # TODO: Implement the leapfrog integrator
    #
    # Hint 1: You need the gradient of log_prob_fn. Use jax.grad to get it:
    #         grad_log_prob = grad(log_prob_fn)
    #
    # Hint 2: The half-step / full-step / half-step pattern means:
    #   for each step:
    #       rho = rho + (epsilon/2) * grad_log_prob(theta)   # half step momentum
    #       theta = theta + epsilon * rho                     # full step position
    #       rho = rho + (epsilon/2) * grad_log_prob(theta)   # half step momentum

    theta_new = ...
    rho_new = ...

    return theta_new, rho_new


def hmc_step(key, theta, log_prob_fn, epsilon, L):
    """Single HMC transition.

    The HMC algorithm:
        1. Sample fresh momentum: rho ~ N(0, I)
        2. Compute initial Hamiltonian: H = -log_prob(theta) + 0.5 * sum(rho^2)
        3. Run leapfrog to get proposal (theta_prop, rho_prop)
        4. Compute proposed Hamiltonian
        5. Accept with probability min(1, exp(-delta_H))

    Args:
        key: JAX random key (used for sampling momentum and accept/reject)
        theta: Current position, shape (D,)
        log_prob_fn: Function that returns log probability
        epsilon: Leapfrog step size
        L: Number of leapfrog steps

    Returns:
        theta_new: New position (may be same as old if rejected)
        accepted: Boolean indicating if proposal was accepted
    """
    # Split the key: key1 for sampling momentum, key2 for accept/reject
    # JAX requires splitting keys to get independent random numbers
    key1, key2 = random.split(key)

    # TODO: Implement HMC step
    #
    # Step 1: Sample momentum from N(0, I)
    #         Use: random.normal(key1, shape=theta.shape)
    # rho = ...
    #
    # Step 2: Compute initial Hamiltonian H = U(theta) + K(rho)
    #         where U(theta) = -log_prob(theta) and K(rho) = 0.5 * sum(rho^2)
    # H_init = ...
    #
    # Step 3: Run leapfrog integration
    # theta_prop, rho_prop = leapfrog(...)
    #
    # Step 4: Compute proposed Hamiltonian
    # H_prop = ...
    #
    # Step 5: Metropolis accept/reject
    #         delta_H = H_prop - H_init
    #         Accept if log(u) < -delta_H where u ~ Uniform(0,1)
    #         Use: random.uniform(key2) to sample u
    # accepted = ...
    #
    # Step 6: Return new theta (proposal if accepted, current if rejected)
    #         Use: jnp.where(accepted, theta_prop, theta)
    # theta_new = ...

    theta_new = ...
    accepted = ...

    return theta_new, accepted
