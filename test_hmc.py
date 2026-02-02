"""
Autograding tests for Lab 2: Hamiltonian Monte Carlo

These tests verify the student's implementation of leapfrog and hmc_step.
"""

import jax.numpy as jnp
import jax.random as jr

from hmc import leapfrog, hmc_step


def run_hmc_sampler(key, log_prob_fn, theta_init, n_samples, epsilon, L):
    """Helper to run HMC sampler for testing."""
    from jax import jit, lax

    keys = jr.split(key, n_samples)

    @jit
    def jit_hmc_step(key, theta):
        return hmc_step(key, theta, log_prob_fn, epsilon, L)

    def scan_fn(theta, key):
        new_theta, accepted = jit_hmc_step(key, theta)
        return new_theta, (new_theta, accepted)

    _, (samples, accepted) = lax.scan(scan_fn, theta_init, keys)
    acceptance_rate = accepted.mean()

    return samples, acceptance_rate


class TestLeapfrog:
    """Tests for the leapfrog integrator."""

    def test_leapfrog_energy_conservation(self):
        """Leapfrog should approximately conserve energy on a simple Gaussian."""

        def simple_log_prob(theta):
            return -0.5 * jnp.sum(theta**2)

        theta0 = jnp.array([1.0, 0.5])
        rho0 = jnp.array([0.5, -0.3])

        # Initial energy
        H0 = -simple_log_prob(theta0) + 0.5 * jnp.sum(rho0**2)

        # Run leapfrog
        theta_new, rho_new = leapfrog(theta0, rho0, simple_log_prob, epsilon=0.1, L=50)

        # Final energy
        H1 = -simple_log_prob(theta_new) + 0.5 * jnp.sum(rho_new**2)

        energy_error = jnp.abs(H1 - H0)
        assert energy_error < 0.01, f"Energy not conserved: error = {energy_error:.6f}"

    def test_leapfrog_moves_position(self):
        """Leapfrog should move the position."""

        def simple_log_prob(theta):
            return -0.5 * jnp.sum(theta**2)

        theta0 = jnp.array([1.0, 0.5])
        rho0 = jnp.array([0.5, -0.3])

        theta_new, rho_new = leapfrog(theta0, rho0, simple_log_prob, epsilon=0.1, L=10)

        # Position should have changed
        assert not jnp.allclose(theta0, theta_new), "Leapfrog did not move position"

    def test_leapfrog_reversibility(self):
        """Leapfrog should be reversible (run forward then backward with negated momentum)."""

        def simple_log_prob(theta):
            return -0.5 * jnp.sum(theta**2)

        theta0 = jnp.array([1.0, 0.5])
        rho0 = jnp.array([0.5, -0.3])

        # Forward
        theta1, rho1 = leapfrog(theta0, rho0, simple_log_prob, epsilon=0.1, L=20)

        # Backward (negate momentum)
        theta2, rho2 = leapfrog(theta1, -rho1, simple_log_prob, epsilon=0.1, L=20)

        # Should return to start
        assert jnp.allclose(theta0, theta2, atol=1e-5), "Leapfrog is not reversible"


class TestHMCStep:
    """Tests for the HMC step function."""

    def test_hmc_returns_correct_shapes(self):
        """HMC step should return theta of correct shape and boolean accepted."""

        def simple_log_prob(theta):
            return -0.5 * jnp.sum(theta**2)

        key = jr.PRNGKey(0)
        theta = jnp.array([1.0, 0.5])

        theta_new, accepted = hmc_step(key, theta, simple_log_prob, epsilon=0.1, L=10)

        assert theta_new.shape == theta.shape, f"Wrong shape: {theta_new.shape}"
        assert accepted.dtype == jnp.bool_, f"accepted should be boolean, got {accepted.dtype}"

    def test_hmc_samples_correct_mean(self):
        """HMC should produce samples with correct mean on a simple Gaussian."""
        true_mean = jnp.array([1.0, -0.5])

        def gaussian_log_prob(theta):
            return -0.5 * jnp.sum((theta - true_mean) ** 2)

        key = jr.PRNGKey(123)
        samples, acc_rate = run_hmc_sampler(
            key, gaussian_log_prob, theta_init=jnp.zeros(2), n_samples=2000, epsilon=0.2, L=10
        )

        # Check mean (discard first 500 as burn-in)
        sample_mean = samples[500:].mean(axis=0)
        mean_error = jnp.max(jnp.abs(sample_mean - true_mean))

        assert mean_error < 0.15, f"Sample mean {sample_mean} too far from true mean {true_mean}"
        assert acc_rate > 0.5, f"Acceptance rate {acc_rate:.2%} too low"

    def test_hmc_acceptance_rate_reasonable(self):
        """HMC should have reasonable acceptance rate with good hyperparameters."""

        def simple_log_prob(theta):
            return -0.5 * jnp.sum(theta**2)

        key = jr.PRNGKey(42)
        samples, acc_rate = run_hmc_sampler(
            key, simple_log_prob, theta_init=jnp.zeros(2), n_samples=500, epsilon=0.1, L=20
        )

        assert acc_rate > 0.6, f"Acceptance rate {acc_rate:.2%} too low"
        assert acc_rate < 1.0, "Acceptance rate should not be 100% (might indicate bug)"
