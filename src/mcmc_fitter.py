import sys
import os
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from typing import Tuple

# 1. Path Setup (Cross-Repository Imports)
# We go up two levels from 'epidemic-parameter-estimation/src/' to the main workspace
current_dir = os.path.dirname(os.path.abspath(__file__))
toolkit_path = os.path.abspath(os.path.join(current_dir, '../../ode-modeling-toolkit'))
malaria_path = os.path.abspath(os.path.join(current_dir, '../../malaria-mathematical-modeling'))

sys.path.append(toolkit_path)
sys.path.append(malaria_path)

# Import your actual flagship model
from src.model import MalariaReplacementModel

# 2. Forward Model Wrapper
def solve_malaria(t_eval: np.ndarray, beta_hv: float, beta_vh: float) -> np.ndarray:
    """Runs the Malaria Replacement Model and returns the Infected Humans (I_h) curve."""
    parameters = {
        'Lambda_h': 10.0, 'mu_h': 0.00004, 'nu_h': 0.1, 'gamma_h': 0.05, 'delta_h': 0.001,
        'b': 0.5, 'beta_hv': beta_hv, 'beta_vh': beta_vh,
        'Lambda_v': 1000.0, 'mu_v': 0.07, 'nu_v': 0.08,
        'Lambda_w': 200.0, 'mu_w': 0.08, 'c': 0.0001, 'u': 50.0 
    }
    initial_states = [10000.0, 0.0, 100.0, 0.0, 50000.0, 0.0, 500.0, 0.0]
    
    model = MalariaReplacementModel(parameters, initial_states)
    sol = model.solve(t_span=(t_eval[0], t_eval[-1]), t_eval=t_eval)
    
    # State Variables: [Sh, Eh, Ih, Rh, Sv, Ev, Iv, W]
    # We want to fit against the Infected Humans (I_h) curve, which is index 2.
    return sol.y[2]

# 3. Bayesian Formulation
def log_prior(theta: Tuple[float, float]) -> float:
    """Uniform priors for transmission probabilities."""
    beta_hv, beta_vh = theta
    # Both transmission probabilities must logically remain between 0 and 1
    if 0.0 < beta_hv < 1.0 and 0.0 < beta_vh < 1.0:
        return 0.0
    return -np.inf

def log_likelihood(theta: Tuple[float, float], t: np.ndarray, y_data: np.ndarray, y_err: float) -> float:
    """Evaluates how well the proposed transmission rates reproduce the observed data."""
    beta_hv, beta_vh = theta
    try:
        model_y = solve_malaria(t, beta_hv, beta_vh)
    except Exception:
        # If the ODE solver fails due to impossible parameter combinations, reject this step
        return -np.inf
        
    return -0.5 * np.sum(((y_data - model_y) / y_err) ** 2 + np.log(2 * np.pi * y_err ** 2))

def log_probability(theta: Tuple[float, float], t: np.ndarray, y_data: np.ndarray, y_err: float) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y_data, y_err)

def main():
    print("--- Initializing Malaria MCMC Parameter Estimation ---")
    
    # 1. Generate Synthetic Data
    np.random.seed(42)
    t_data = np.linspace(0, 100, 20) # 100 days of data, measured every 5 days
    true_beta_hv, true_beta_vh = 0.3, 0.4
    
    print("Generating synthetic epidemic data using true parameters...")
    true_infected = solve_malaria(t_data, true_beta_hv, true_beta_vh)
    
    # Add realistic observation noise (e.g., standard deviation of 15 cases per reading)
    y_err = 15.0 
    y_data = true_infected + y_err * np.random.randn(len(t_data))
    y_data = np.maximum(y_data, 0) # Reported cases cannot be negative

    # 2. Setup emcee Sampler
    ndim = 2
    nwalkers = 16  # Reduced to 16 for faster computation on the 8-variable ODE system
    nsteps = 400   # 400 steps is sufficient for burn-in demonstration
    
    # Initialize walkers around a rough guess (0.5, 0.5)
    initial_guess = np.array([0.5, 0.5])
    pos = initial_guess + 1e-2 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t_data, y_data, y_err))
    
    # 3. Run MCMC
    print(f"Running {nwalkers} walkers for {nsteps} steps.")
    print("Note: Evaluating an 8-compartment ODE thousands of times takes a moment...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # 4. Results
    samples = sampler.get_chain(discard=100, flat=True)
    beta_hv_est, beta_vh_est = np.median(samples, axis=0)
    
    print(f"\nTrue Parameters: beta_hv={true_beta_hv}, beta_vh={true_beta_vh}")
    print(f"MCMC Estimates:  beta_hv={beta_hv_est:.3f}, beta_vh={beta_vh_est:.3f}")
    
    # 5. Plot
    print("\nGenerating Posterior Distribution Corner Plot...")
    fig = corner.corner(
        samples, 
        labels=["$\\beta_{hv}$", "$\\beta_{vh}$"], 
        truths=[true_beta_hv, true_beta_vh],
        truth_color="#e74c3c",
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    plt.show()

if __name__ == "__main__":
    main()