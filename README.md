# **invert** - A high-level M/EEG Python library for EEG inverse solutions

This package contains various approaches to solve the M/EEG inverse problems. It
integrates with the [mne-python](https://mne.tools) framework.

## Inverse Solutions

### Minimum-Norm based
* Minimum Norm Estimates
* Weighted Minimum Norm Estimates
* Dynamic Statistical Parametric Mapping (dSPM)
* Minimum Current Estimates (FISTA solver)
* L1-L2 Optimization

### LORETA Family
* Low-Resolution Tomography (LORETA)
* Standardized Low-Resolution Tomography (sLORETA)
* Exact Low-Resolution Tomography (eLORETA)
* S-MAP

### Weighted Resolution Optimization
* Backus-Gilbert
* Local Autoregressive Average (LAURA)

### Bayesian Family
* Gamma - Maximum A Posteriori (Gamma-MAP)
* Source - Maximum A Posteriori (Source-MAP)
* Bayesian Compressed Sensing (BCS)
* Champagne
* Multiple Sparse Priors (MSP)
* Bayesian LORETA
* Bayesian MNE
* Bayesian Beamformer

### Greedy Algorithms / Matching Pursuit
* Orthogonal Matching Pursuit (OMP)
* Simultaneous Orthogonal Matching Pursuit (SOMP)
* Compressed Sampling Matching Pursuit (CoSaMP)
* Reduce Multi-Measurement-Vector and Boost (ReMBo)
* Subspace Pursuit (SP)

### Smooth Greedy Algorithms / Matching Pursuit
* Smooth Orthogonal Matching Pursuit (SMP)
* Smooth Simultaneous Orthogonal Matching Pursuit (SSMP)
* Smooth Subspace Simultaneous Orthogonal Matching Pursuit (SubSMP)

### Subspace Techniques
* Multiple Signal Classification (MUSIC)
* Recursively Applied MUSIC (RAP-MUSIC)
* Truncated RAP-Music (TRAP-MUSIC)

### Artificial Neural Networks
* Convolutional Neural Network for Distributed Dipole Solutions (ConvDip)

### Beamforming
* Minimum Variance Adaptive Beamformer (MVAB)

### Other
* EPIFOCUS


## Adapters
Adapters are methods that optimize an already calculated inverse solution, e.g.,
by enforcing temporal constraints or by iteratively promoting sparsity.

* Temporal contextualization using LSTMs (C-MNE) [10]
* Source re-weighting (FOCUSS) [11]
* Spatio-Temporal Matching Pursuit for Contextualization (STAMP-C)

# References

[10] Dinh, C., Samuelsson, J. G., Hunold, A., Hämäläinen, M. S., & Khan, S.
(2021). Contextual MEG and EEG source estimates using spatiotemporal LSTM
networks. Frontiers in neuroscience, 15, 552666. 

[11] Gorodnitsky, I. F., & Rao, B. D. (1997). Sparse signal reconstruction from
limited data using FOCUSS: A re-weighted minimum norm algorithm. IEEE
Transactions in signal processing, 45(3), 600-616.
