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


def test_hmc_produces_samples():
    """HMC should produce varying samples over multiple steps."""

    def simple_log_prob(theta):
        return -0.5 * jnp.sum(theta**2)

    key = jr.PRNGKey(42)
    theta = jnp.zeros(2)
    samples = [theta]

    for i in range(20):
        key, subkey = jr.split(key)
        theta, _ = hmc_step(subkey, theta, simple_log_prob, epsilon=0.2, L=10)
        samples.append(theta)

    samples = jnp.stack(samples)
    # Check that samples have some variance (not stuck)
    assert jnp.std(samples) > 0.1, "HMC appears to be stuck"
