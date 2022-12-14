Solvers
=======

This is the comprehensive list of all available solvers. Each solver has its own
class

:doc:`minimum_norm`
===================
- Minimum Norm Estimates (MNE)
- Weighted Minimum Norm Estimates
- Dynamic Statistical Parametric Mapping (dSPM)
- Minimum Current Estimates (FISTA solver)
- L1-L2 Optimization

:doc:`loreta`
=============
- Low-Resolution Tomography (LORETA)
- Standardized Low-Resolution Tomography (sLORETA)
- Exact Low-Resolution Tomography (eLORETA)
- S-MAP

:doc:`wrop`
================================
- Backus-Gilbert
- Local Autoregressive Average (LAURA)

:doc:`bayes`
===============
- Gamma - Maximum A Posteriori (Gamma-MAP)
- Gamma - Maximum A Posteriori using Multiple Sparse Priors (Source-MAP-MSP)
- Source - Maximum A Posteriori (Source-MAP)
- Source - Maximum A Posteriori using Multiple Sparse Priors (Source-MAP-MSP)
- Bayesian Compressed Sensing (BCS)
- Champagne
- Low SNR Champagne


:doc:`matching_pursuit`
==========
- Orthogonal Matching Pursuit (OMP)
- Simultaneous Orthogonal Matching Pursuit (SOMP)
- Compressed Sampling Matching Pursuit (CoSaMP)
- Reduce Multi-Measurement-Vector and Boost (ReMBo)
- Subspace Pursuit (SP)

:doc:`smooth_matching_pursuit`
=========
- Smooth Orthogonal Matching Pursuit (SMP)
- Smooth Simultaneous Orthogonal Matching Pursuit (SSMP)
- Smooth Subspace Simultaneous Orthogonal Matching Pursuit (SubSMP)

:doc:`music`
==========
- Multiple Signal Classification (MUSIC)
- Recursively Applied MUSIC (RAP-MUSIC)
- Truncated RAP-Music (TRAP-MUSIC)
- Smooth Truncated RAP-Music (JAZZ-MUSIC)

:doc:`artificial_neural_networks`
==========
- Convolutional Neural Network for Spatio-Temporal Inverse Solution
- Long-Short Term Memory Network for Spatio-Temporal Inverse Solution
- Fully-Connected Neural Network for Inverse Solution at single time instances
- Covariance-based Convolutional Neural Network for Spatio-Temporal Inverse Solution

:doc:`beamformer`
=================
- Minimum Variance Adaptive (MVAB) Beamformer
- Linearly Constrained Minimum Variance (LCMV) Beamformer
- Standardized Minimum Variance (SMV) Beamformer
- Weight-normalized Minimum Variance (WNMV) Beamformer
- Higher-Order Minimum Variance (HOCMV) Beamformer
- Eigenspace-Based Minimum Variance (ESMV) Beamformer
- Multiple Constrained Minimum Variance (MCMV) Beamformer
- Eigenspace-Based Multiple Constrained Minimum Variance (ESMCMV) Beamformer
- Reciprocal Phase Shift Invariant Imaging of Coherent Sources (ReciPSIICOS)
  Beamformer
- Synthetic Aperture Magnetometry (SAM)

Other
=====
- :doc:`epifocus`

.. toctree::
    :maxdepth: 2
    :caption: Full List of Solvers

    minimum_norm
    loreta
    wrop
    bayes
    matching_pursuit
    smooth_matching_pursuit
    music
    artificial_neural_networks
    beamformer
    epifocus