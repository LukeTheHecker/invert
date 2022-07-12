# **invert** - A high-level M/EEG Python library for all kinds of EEG inverse solutions

This package contains various approaches to solve the M/EEG inverse problems. It
integrates with the [mne-python](https://mne.tools) framework.

## Inverse Solutions

Minimum-Norm based:
* Minimum Norm Estimates
* Weighted Minimum Norm Estimates
* Dynamic Statistical Parametric Mapping (dSPM)

LORETA Family:
* Low-Resolution Tomography (LORETA)
* Standardized Low-Resolution Tomography (sLORETA)
* Exact Low-Resolution Tomography (eLORETA)

Weighted Resolution Optimization:
* Backus-Gilbert
* Local Autoregressive Average (LAURA)

Bayesian Family:
* Multiple Sparse Priors
* Bayesian LORETA
* Bayesian MNE
* Bayesian Beamformer

Other:
* S-MAP

## Adapters
Adapters are methods that optimize an already calculated inverse solution, e.g.,
by enforcing temporal constraints or by iteratively promoting sparsity.

* Temporal contextualization using the approach described in [10].
* FOCUSS - decreases blurring in the inverse solutions [11].

# References
[10] Dinh, C., Samuelsson, J. G., Hunold, A., Hämäläinen, M. S., & Khan, S.
(2021). Contextual MEG and EEG source estimates using spatiotemporal LSTM
networks. Frontiers in neuroscience, 15, 552666. 

[11] Gorodnitsky, I. F., & Rao, B. D. (1997). Sparse signal reconstruction from
limited data using FOCUSS: A re-weighted minimum norm algorithm. IEEE
Transactions in signal processing, 45(3), 600-616.
