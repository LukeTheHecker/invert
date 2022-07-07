# Architecture


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


- re-think how the lstm is designed and whether ideas from cmne could be adapted
- consider dropout on input layer for bad-channel robustness
