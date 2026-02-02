"""
Autograding tests for Lab 2: Hamiltonian Monte Carlo
"""

import jax.numpy as jnp
import jax.random as jr

from hmc import leapfrog, hmc_step


def test_leapfrog_energy_conservation():
    """Leapfrog should approximately conserve energy."""

    def simple_log_prob(theta):
        return -0.5 * jnp.sum(theta**2)

    theta0 = jnp.array([1.0, 0.5])
    rho0 = jnp.array([0.5, -0.3])

    H0 = -simple_log_prob(theta0) + 0.5 * jnp.sum(rho0**2)
    theta_new, rho_new = leapfrog(theta0, rho0, simple_log_prob, epsilon=0.1, L=50)
    H1 = -simple_log_prob(theta_new) + 0.5 * jnp.sum(rho_new**2)

    assert jnp.abs(H1 - H0) < 0.01, "Energy not conserved"


def test_leapfrog_moves_position():
    """Leapfrog should move the position."""

    def simple_log_prob(theta):
        return -0.5 * jnp.sum(theta**2)

    theta0 = jnp.array([1.0, 0.5])
    rho0 = jnp.array([0.5, -0.3])

    theta_new, _ = leapfrog(theta0, rho0, simple_log_prob, epsilon=0.1, L=10)

    assert not jnp.allclose(theta0, theta_new), "Leapfrog did not move position"


def test_hmc_returns_correct_shape():
    """HMC step should return theta of correct shape."""

    def simple_log_prob(theta):
        return -0.5 * jnp.sum(theta**2)

    key = jr.PRNGKey(0)
    theta = jnp.array([1.0, 0.5])

    theta_new, accepted = hmc_step(key, theta, simple_log_prob, epsilon=0.1, L=10)

    assert theta_new.shape == theta.shape, f"Wrong shape: {theta_new.shape}"


def test_hmc_samples_correct_mean():
    """HMC should produce samples with correct mean on a simple Gaussian."""
    from jax import jit, lax

    true_mean = jnp.array([1.0, -0.5])

    def gaussian_log_prob(theta):
        return -0.5 * jnp.sum((theta - true_mean) ** 2)

    key = jr.PRNGKey(123)
    keys = jr.split(key, 2000)
    theta_init = jnp.zeros(2)

    @jit
    def jit_hmc_step(key, theta):
        return hmc_step(key, theta, gaussian_log_prob, epsilon=0.2, L=10)

    def scan_fn(theta, key):
        new_theta, accepted = jit_hmc_step(key, theta)
        return new_theta, new_theta

    _, samples = lax.scan(scan_fn, theta_init, keys)

    # Check mean (discard first 500 as burn-in)
    sample_mean = samples[500:].mean(axis=0)
    mean_error = jnp.max(jnp.abs(sample_mean - true_mean))

    assert mean_error < 0.15, f"Sample mean {sample_mean} too far from true mean {true_mean}"
