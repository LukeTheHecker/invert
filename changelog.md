# Changelog

## 0.0.5 ???
* removed automated regularisation from sloreta due to unexpected behaviors
* refactored FLEX-MUSIC with minor speed improvements
* fixed an error in the final estimation of the current source density with FLEX-MUSIC

## 0.0.4 30.12.2022
* Change in FLEX-MUSIC: use selected dipoles/ smooth patches as source covariance during WMNE inversion
  * before: indices were selected as full flat priors for WMNE inversion
* Fixed error with the sLORETA formula

## 0.0.3 28.12.2022
* Added rank-reduction option to invert.BaseSolver class
* Set rank_reduction as default for all Beamforming approaches for better results

## 0.0.2 20.12.2022
* Added PyQt5 to the package dependencies

## 0.0.1 20.12.2022
* Release of the invert package.