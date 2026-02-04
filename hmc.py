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
    grad_log_prob = grad(log_prob_fn)

    # keeping function pure
    theta_copy = jnp.copy(theta)
    rho_copy = jnp.copy(rho)

    # implement each step
    for _ in range(L):
        # half step momentum
        # feels the 'gravity' of the slope it is sitting on
        rho_copy += (epsilon / 2) * grad_log_prob(theta_copy)

        # full step position
        # 'moving' the actual marble
        theta_copy += epsilon * rho_copy

        # half step momentum
        # recenters the 'velocity' so that the simulation stays table,
        # basically recalculates the 'gravity' at the new spot
        rho_copy += (epsilon / 2) * grad_log_prob(theta_copy)

    # return tuple of the new steps
    return theta_copy, rho_copy

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

    # random momentum in a normal distribution
    rho = random.normal(key1, shape=theta.shape)

    # The dot product of a vector with itself dot(rho, rho) is the same as the sum of its squares, which is rho^Trho
    h_init = -log_prob_fn(theta) + 0.5 * jnp.dot(rho, rho)

    # theta_prop is where the particle lands; rho_prop is its final speed
    theta_prop, rho_prop = leapfrog(theta, rho, log_prob_fn, epsilon, L)

    # final Hamiltonian
    h_final = -log_prob_fn(theta_prop) + 0.5 * jnp.dot(rho_prop, rho_prop)

    # delta_H is the change in energy. Ideally, it's 0.
    # we want to energy to be conserved
    delta_h = h_final - h_init
    
    # We accept if the new energy is lower, or with prob exp(-delta_H) if higher
    # when energy is lower it means we ar emoving to a higher probability area
    # a 'deeper' part of the valley
    accept_prob = jnp.exp(-delta_h)
    u = random.uniform(key2)
    
    accepted = u < accept_prob
    theta_new = jnp.where(accepted, theta_prop, theta)

    return theta_new, accepted
