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

