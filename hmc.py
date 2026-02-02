"""
Lab 2: Hamiltonian Monte Carlo Implementation
"""

import jax.numpy as jnp
from jax import grad, random


def leapfrog(theta, rho, log_prob_fn, epsilon, L):
    """Run L steps of leapfrog integration.

    Each step:
        1. Half step momentum: rho += (epsilon/2) * grad_log_prob(theta)
        2. Full step position: theta += epsilon * rho
        3. Half step momentum: rho += (epsilon/2) * grad_log_prob(theta)

    Args:
        theta: Position, shape (D,)
        rho: Momentum, shape (D,)
        log_prob_fn: Log probability function
        epsilon: Step size
        L: Number of steps

    Returns:
        theta_new, rho_new
    """
    # TODO: Implement leapfrog
    # Hint: Use grad(log_prob_fn) to get the gradient function

    ...


def hmc_step(key, theta, log_prob_fn, epsilon, L):
    """Single HMC transition.

    1. Sample momentum: rho ~ N(0, I)
    2. Compute initial Hamiltonian: H = -log_prob(theta) + 0.5 * rho^T @ rho
    3. Run leapfrog to get proposal
    4. Accept with probability min(1, exp(-delta_H))

    Args:
        key: JAX random key
        theta: Current position, shape (D,)
        log_prob_fn: Log probability function
        epsilon: Leapfrog step size
        L: Number of leapfrog steps

    Returns:
        theta_new: New position (proposal if accepted, current if rejected)
        accepted: Boolean
    """
    # Split key for independent random numbers (one for momentum, one for accept/reject)
    key1, key2 = random.split(key)

    # TODO: Implement HMC step
    # Hints for JAX random:
    #   random.normal(key1, shape=theta.shape)  # sample momentum
    #   random.uniform(key2)                    # sample for accept/reject

    ...
