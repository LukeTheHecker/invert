# Architecture

## make_inverse_operator()

Solvers that have a generic operator:

MNE, wMNE, dSPM, LORETA, sLORETA, eLORETA, LAURA, S-MAP, BG,   esinet

Solvers that don't:
MSP, bLORETA, bBeamformer, bMNE, LUCAS, 

### Required inputs to make inverse operator

MNE, wMNE, sLORETA: 
leadfield, alpha=0.001, noise_cov=None

eLORETA:
leadfield, alpha=0.001, stop_crit=0.005, noise_cov=None


dSPM:
leadfield, alpha=0.001, noise_cov=None, source_cov=None

LORETA, S-MAP:
leadfield, adjacency, alpha=0.001,

LAURA:
leadfield, adjacency, pos, alpha=200, drop_off=2, noise_cov=None

backus-gilbert:
leadfield, pos, 

esinet:
fwd, info, 





## post-processors

* focuss(inverse_operator, M, leadfield, alpha)
* contextualize(stc, leadfield, **kwargs)
* contextualize_bd(stc, leadfield, **kwargs)


## notes..

for the package:

- think about structure of folders/ modules
	- think about functions and classes:
		- make_inverse_operator + apply_inverse_operator
		- inverse_class().fit(fwd).apply(data)  # or
		  inverse_class().fit(data, fwd).apply(data)	
- from CMNE repo implement the past-sample average method
- CMNE amplitudes are way off - how did they meant to do it in the paper/repo?
- implement st-map, wrop, Beamformer, MUSIC, BESA
- implement iterative focusing (for S-MAP only??), shrinking, SSLOFO, ALF,
- for each solver, figure out how to use:
  - noise_cov
  - source_cov (i.e. priors)