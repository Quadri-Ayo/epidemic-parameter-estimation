# Epidemiological Parameter Estimation and Uncertainty Analysis

## Overview

This project investigates computational methods for estimating unknown parameters in mathematical epidemiological models from observed or simulated epidemic data.

In a mathematical model, parameters such as transmission and recovery rates determine the behaviour of the system. In practical applications, however, these quantities are rarely known exactly.

The project therefore treats parameter estimation as an inverse problem:

$$
\text{Observed data}
\longrightarrow
\text{Model}
\longrightarrow
\text{Parameter inference}.
$$

---

## Research Question

**How accurately can epidemiological model parameters be recovered from incomplete and noisy observations?**

The project investigates this question using computational parameter-estimation methods and uncertainty analysis.

---

## Mathematical Framework
Consider an epidemiological model represented by a system of ordinary differential equations:

$$\frac{d\mathbf{x}}{dt} = f(t, \mathbf{x}, \theta)$$

where:
*   $\mathbf{x}(t)$ is the state vector.
*   $f$ represents the model dynamics.
*   $\theta$ represents unknown parameters.

Given discrete observations

$$
y_1,y_2,\ldots,y_n,
$$

the objective is to infer plausible values of \(\theta\).

---

## Methodology

The computational workflow consists of:

```text
Generate / obtain observations
             ↓
Specify epidemiological model
             ↓
Define unknown parameters
             ↓
Define likelihood / objective
             ↓
Estimate parameters
             ↓
Quantify uncertainty
             ↓
Compare model predictions
             ↓
Assess parameter recovery
```

---

## Parameter Estimation

The project investigates computational approaches for estimating parameters such as:

$$
\beta = \text{transmission parameter},
$$

$$
\gamma = \text{recovery parameter}.
$$

Depending on the experiment, estimation may be formulated using optimization or probabilistic inference.

---

## Bayesian Inference

Where Bayesian inference is used, the posterior distribution is represented as

$$
p(\theta\mid y)
\propto
p(y\mid\theta)p(\theta),
$$

where:

* \(p(\theta\mid y)\) is the posterior distribution;
* \(p(y\mid\theta)\) is the likelihood;
* \(p(\theta)\) is the prior distribution.

Sampling-based methods are used to investigate the resulting parameter distributions.

---

## Uncertainty Quantification

Rather than reporting only a single parameter estimate, the project investigates uncertainty in the estimated parameters.

This allows model predictions to be expressed in terms of plausible parameter ranges rather than treating estimated values as exact.

Where appropriate, results are presented using:

* posterior distributions;
* credible intervals;
* parameter traces;
* predictive intervals;
* and comparison between estimated and known parameters.

---

## Computational Tools

The implementation uses Python and relevant scientific-computing libraries, including:

* NumPy
* SciPy
* Pandas
* Matplotlib
* Jupyter
* `emcee` where Bayesian sampling is employed

---

## Research Experiment

The initial experiments use controlled or synthetic epidemic data to evaluate whether the estimation procedure can recover known parameters.

This provides a controlled environment for testing the methodology before applying it to real-world epidemiological observations.

---

## Model Validation

Parameter-estimation results are evaluated by comparing:

$$
\text{Observed data}
\quad\text{vs.}\quad
\text{Model predictions}.
$$

Where synthetic data are used, recovered parameter values can additionally be compared with the parameters used to generate the data.

---

## Limitations

Parameter identifiability, observation noise, model misspecification, limited observations, and parameter correlations can affect inference.

Consequently, successful numerical estimation does not necessarily imply that the estimated parameters are uniquely identifiable from real-world data.

---

## Reproducibility

The project is version-controlled through GitHub, with computational experiments documented through source code and notebooks.

The objective is to allow another researcher to reproduce the parameter-estimation experiments from the documented model, data-generation procedure, and computational settings.

---

## Connection to Mathematical Modelling

This project complements the broader malaria-modelling research portfolio by addressing the inverse side of mathematical modelling.

The relationship can be summarized as:

$$
\boxed{
\text{Mathematical Model}
\rightarrow
\text{Simulation}
\rightarrow
\text{Data}
\rightarrow
\text{Parameter Inference}
}
$$

Together, these projects form a computational workflow for developing, analysing, and calibrating mathematical models of biological systems.

> **Status:** Active independent research project.
