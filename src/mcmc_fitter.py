import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple

# 1. Define the Forward Model (Simple SIR for demonstration)
def sir_derivs(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> list:
    """Standard SIR ODE system."""
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def solve_sir(t_eval: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Integrates the SIR model over the given time points."""
    N = 1000.0
    y0 = [990.0, 10.0, 0.0] # Initial conditions
    sol = solve_ivp(sir_derivs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, args=(beta, gamma, N))
    return sol.y[1] # Return only the Infected curve for fitting

# 2. Bayesian Formulation (Prior, Likelihood, Posterior)
def log_prior(theta: Tuple[float, float]) -> float:
    """
    Uniform prior distributions for the parameters.
    beta ~ U(0, 1)
    gamma ~ U(0, 1)
    """
    beta, gamma = theta
    if 0.0 < beta < 1.0 and 0.0 < gamma < 1.0:
        return 0.0
    return -np.inf # Outside bounds is impossible

def log_likelihood(theta: Tuple[float, float], t: np.ndarray, y_data: np.ndarray, y_err: float) -> float:
    """
    Calculates how well the model output matches the data given parameters 'theta'.
    Assumes Gaussian noise.
    """
    beta, gamma = theta
    model_y = solve_sir(t, beta, gamma)
    # Log-likelihood of a Gaussian error distribution
    return -0.5 * np.sum(((y_data - model_y) / y_err) ** 2 + np.log(2 * np.pi * y_err ** 2))

def log_probability(theta: Tuple[float, float], t: np.ndarray, y_data: np.ndarray, y_err: float) -> float:
    """Bayes' Theorem: log(Posterior) = log(Prior) + log(Likelihood)"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y_data, y_err)

def main():
    print("--- Initializing MCMC Parameter Estimation ---")
    
    # 1. Generate Synthetic 'Real-World' Data (True parameters: beta=0.3, gamma=0.1)
    np.random.seed(42)
    t_data = np.linspace(0, 50, 20)
    true_beta, true_gamma = 0.3, 0.1
    true_infected = solve_sir(t_data, true_beta, true_gamma)
    
    # Add Gaussian noise to simulate messy public health reporting
    y_err = 5.0 
    y_data = true_infected + y_err * np.random.randn(len(t_data))
    y_data = np.maximum(y_data, 0) # Cases cannot be negative

    # 2. Setup the emcee Sampler
    ndim = 2          # Number of parameters (beta, gamma)
    nwalkers = 32     # Number of independent MCMC chains
    nsteps = 1000     # Number of steps per walker
    
    # Initialize walkers in a tight ball around a random guess
    initial_guess = np.array([0.5, 0.5])
    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t_data, y_data, y_err))
    
    # 3. Run the MCMC Chain
    print(f"Running {nwalkers} walkers for {nsteps} steps...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # 4. Analyze Results
    # Discard the first 200 steps as 'burn-in' (the time it takes walkers to find the high-probability region)
    samples = sampler.get_chain(discard=200, flat=True)
    
    beta_mcmc, gamma_mcmc = np.median(samples, axis=0)
    print(f"\nTrue Parameters: beta={true_beta}, gamma={true_gamma}")
    print(f"MCMC Estimates:  beta={beta_mcmc:.3f}, gamma={gamma_mcmc:.3f}")
    
    # 5. Plot the Corner Plot (Posterior Distributions)
    print("\nGenerating Posterior Distribution Corner Plot...")
    fig = corner.corner(
        samples, 
        labels=["$\\beta$", "$\\gamma$"], 
        truths=[true_beta, true_gamma],
        truth_color="#e74c3c",
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    plt.show()

if __name__ == "__main__":
    main()