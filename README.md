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

### Weighted Resolution Optimization
* Backus-Gilbert
* Local Autoregressive Average (LAURA)

### Bayesian Family
* Gamma - Maximum A Posteriori (Gamma-MAP)
* Gamma - Maximum A Posteriori using Multiple Sparse Priors (Source-MAP-MSP)
* Source - Maximum A Posteriori (Source-MAP)
* Source - Maximum A Posteriori using Multiple Sparse Priors (Source-MAP-MSP)
* Bayesian Compressed Sensing (BCS)
* Champagne
* Low SNR Champagne


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
* Smooth Truncated RAP-Music (JAZZ-MUSIC)

### Artificial Neural Networks
* Convolutional Neural Network for Spatio-Temporal Inverse Solution
* Long-Short Term Memory Network for Spatio-Temporal Inverse Solution
* Fully-Connected Neural Network for Inverse Solution at single time instances
* Covariance-based Convolutional Neural Network for Spatio-Temporal Inverse Solution

### Beamforming
* Minimum Variance Adaptive (MVAB) Beamformer
* Linearly Constrained Minimum Variance (LCMV) Beamformer
* Standardized Minimum Variance (SMV) Beamformer
* Weight-normalized Minimum Variance (WNMV) Beamformer
* Higher-Order Minimum Variance (HOCMV) Beamformer
* Eigenspace-Based Minimum Variance (ESMV) Beamformer
* Multiple Constrained Minimum Variance (MCMV) Beamformer
* Eigenspace-Based Multiple Constrained Minimum Variance (ESMCMV) Beamformer
* Reciprocal Phase Shift Invariant Imaging of Coherent Sources (ReciPSIICOS)
  Beamformer
* Synthetic Aperture Magnetometry (SAM)

### Other
* EPIFOCUS
* S-MAP

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
