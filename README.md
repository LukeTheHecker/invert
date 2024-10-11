# **invertmeeg** - A high-level M/EEG Python library for EEG inverse solutions :dart:

This package contains various (>50) approaches to solve the M/EEG inverse
problems :leftwards_arrow_with_hook:. It integrates with the [mne-python](https://mne.tools) framework.

Read the [Documentation here!](https://lukethehecker.github.io/invert/)

Install the package from [pypi](https://pypi.org/project/invertmeeg/):

```
pip install invertmeeg
```

To check if the installation works run:

```
python -c 'import invert'
```

To test the package simply run:

```
pytest tests
```

To calculate an inverse solution using *minimum norm estimates* simply type:

```
from invert import Solver

# fwd = ...
# evoked = ...

# Create a Solver instance
solver_name = "MNE"
solver = Solver(solver_name)

# Calculate the inverse operator
solver.make_inverse_operator(fwd)

# Apply the inverse operator to your data
stc = solver.apply_inverse_operator(evoked)

# Plot the resulting source estimate
stc.plot()
```

There are many solvers implemented in the package, and you can find them
[here!](https://lukethehecker.github.io/invert/content/solvers.html)

I am looking for collaborators! If you are interested you can write an :email: to [lukas_hecker@web.de](mailto:lukas_hecker@web.de)

If you use this package and publish results, please cite as:

```
@Misc{invertmeeg2022,
  author =   {{Lukas Hecker}},
  title =    {{invertmeeg}: A high-level M/EEG Python library for EEG inverse solutions.},
  howpublished = {\url{https://github.com/LukeTheHecker/invert}},
  year = {since 2022}
}
```

# List of Algorithms

## Minimum Norm

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Minimum Norm Estimate | "mne" |
| Weighted Minimum Norm Estimate | "wmne" |
| Dynamic Statistical Parametric Mapping | "dspm" |
| Minimum Current Estimate | "l1", "fista", "mce" |
| Minimum L1 Norm GPT | "gpt", "l1 gpt" |
| Minimum L1L2 Norm | "l1l2" |

## LORETA

| Full Solver Name | Abbreviation |
|------------------|--------------|
| LORETA | "lor" |
| sLORETA | "slor" |
| eLORETA | "elor" |

## Other Minimum-norm-like Algorithms

| Full Solver Name | Abbreviation |
|------------------|--------------|
| LAURA | "laura", "laur" |
| Backus-Gilbert | "b-g", "bg" |
| S-MAP | "smap" |

## Bayesian

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Champagne | "champ" |
| Low SNR Champagne | "lsc", "low snr champagne" |
| MacKay Champagne | "mcc", "mackay champagne" |
| Convexity Champagne | "coc", "convexity champagne" |
| Non-Linear Champagne | "champagne-nl" |
| Expectation Maximization Champagne | "emc" |
| Majorization Maximization Champagne | "mmc" |
| Full-Structure Noise | "fun" |
| Heteroscedastic Champagne | "hsc" |
| Gamma-MAP | "gmap" |
| Source-MAP | "source map" |
| Gamma-MAP-MSP | "gamma map msp" |
| Source-MAP-MSP | "source map msp" |

## Beamformers

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Minimum Variance Adaptive Beamformer | "mvab" |
| Linearly Constrained Minimum Variance | "lcmv" |
| Standardized Minimum Variance | "smv" |
| Weight-Normalized Minimum Variance | "wnmv" |
| Higher-Order Covariance Minimum Variance | "hocmv" |
| Eigenspace Scalar Minimum Variance | "esmv" |
| Multiple Constraint Minimum Variance | "mcmv" |
| Higher-Order Covariance Multiple Constraint Minimum Variance | "hocmcmv" |
| Reciprocal PSIICOS | "recipsiicos" |
| Synthetic Aperture Magnetometry | "sam" |
| Empirical Bayesian Beamformer | "ebb" |

## Artificial Neural Networks

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Fully-Connected Network | "fc", "esinet" |
| Covariance CNN | "covcnn", "covnet" |
| Long Short-Term Memory | "lstm" |
| Convolutional Neural Network | "cnn" |

## Matching Pursuit / Compressive Sensing

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Orthogonal Matching Pursuit | "omp" |
| Compressive Sampling Matching Pursuit | "cosamp" |
| Simultaneous Orthogonal Matching Pursuit | "somp" |
| Random Embedding Matching Pursuit | "rembo" |
| Subspace Pursuit | "sp" |
| Structured Subspace Pursuit | "ssp" |
| Subspace Matching Pursuit | "smp" |
| Structured Subspace Matching Pursuit | "ssmp" |
| Subspace-based Subspace Matching Pursuit | "subsmp" |
| Iterative Subspace-based Subspace Matching Pursuit | "isubsmp" |
| Bayesian Compressive Sensing | "bcs" |

## MUSIC/RAP/Subspace

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Multiple Signal Classification | "music" |
| Recursively Applied and Projected MUSIC | "rap-music", "rap" |
| Truncated Recursively Applied and Projected MUSIC | "trap-music", "trap" |
| Flexible RAP-MUSIC | "flex-music", "flex" |
| Flexible Signal Subspace Matching | "flex-ssm" |
| Signal Subspace Matching | "ssm" |
| Flexible Alternating Projections | "flex-ap" |
| Alternating Projections | "ap" |

## Basis Functions

| Full Solver Name | Abbreviation |
|------------------|--------------|
| Geometrically Informed Basis Functions | "gbf" |

## Other

| Full Solver Name | Abbreviation |
|------------------|--------------|
| EPIFOCUS | "epifocus" |