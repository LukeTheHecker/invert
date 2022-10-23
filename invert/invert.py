from time import monotonic
import numpy as np
import mne
from .util import pos_from_forward
import esinet
import matplotlib.pyplot as plt

from invert import solvers
from . import config


def Solver(solver="mne", **kwargs):
    """ Solver class ...
    """

    if solver.lower() == "mne":
        solver_object = solvers.SolverMinimumNorm(**kwargs)
    elif solver.lower() == "wmne":
        solver_object = solvers.SolverWeightedMinimumNorm(**kwargs)
    elif solver.lower() == "dspm":
        solver_object = solvers.SolverDynamicStatisticalParametricMapping(**kwargs)
    elif solver.lower() == "l1l2":
        solver_object = solvers.SolverMinimumL1L2Norm(**kwargs)
    elif solver.lower() == "l1" or solver.lower() == "fista":
        solver_object = solvers.SolverMinimumL1Norm(**kwargs)
    elif solver.lower() == "loreta" or solver.lower() == "lor":
        solver_object = solvers.SolverLORETA(**kwargs)
    elif solver.lower() == "sloreta" or solver.lower() == "slor":
        solver_object = solvers.SolverSLORETA(**kwargs)
    elif solver.lower() == "eloreta" or solver.lower() == "elor":
        solver_object = solvers.SolverELORETA(**kwargs)
    elif solver.lower() == "laura" or solver.lower() == "laur":
        solver_object = solvers.SolverLAURA(**kwargs)
    elif solver.lower() == "backus-gilbert" or solver.lower() == "b-g" or  solver.lower() == "bg":
        solver_object = solvers.SolverBackusGilbert(**kwargs)
    elif solver.lower() == "s-map" or solver.lower() == "smap":
        solver_object = solvers.SolverSMAP(**kwargs)
    elif solver.lower() == "multiple sparse priors" or solver.lower() == "msp":
        inversion_type = "MSP"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type, **kwargs)
    elif solver.lower() == "bayesian loreta" or solver.lower() == "bayesian lor" or solver.lower() == "bloreta" or solver.lower() == "blor":
        inversion_type = "LORETA"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type, **kwargs)
    elif solver.lower() == "bayesian mne" or solver.lower() == "bmne":
        inversion_type = "MNE"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type, **kwargs)
    elif solver.lower() == "bayesian beamformer" or solver.lower() == "bbmf":
        inversion_type = "BMF"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type, **kwargs)
    elif solver.lower() == "bayesian beamformer loreta" or solver.lower() == "bbmf-lor" or solver.lower() == "bmf-lor":
        inversion_type = "BMF-LOR"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type, **kwargs)
    elif solver.lower() == "mvab":
        solver_object = solvers.SolverMVAB(**kwargs)
    elif solver.lower() == "fully-connected" or solver.lower() == "fc" or solver.lower() == "fullyconnected" or solver.lower() == "esinet":
        solver_object = solvers.SolverFullyConnected(**kwargs)
    elif solver.lower() == "lucas":
        solver_object = solvers.SolverLUCAS(**kwargs)
    elif solver.lower() == "champagne" or solver.lower() == "champ":
        solver_object = solvers.SolverChampagne(**kwargs)
    elif solver.lower() == "omp":
        solver_object = solvers.SolverOMP(**kwargs)
    elif solver.lower() == "cosamp":
        solver_object = solvers.SolverCOSAMP(**kwargs)
    elif solver.lower() == "somp":
        solver_object = solvers.SolverSOMP(**kwargs)
    elif solver.lower() == "rembo":
        solver_object = solvers.SolverREMBO(**kwargs)
    elif solver.lower() == "sp":
        solver_object = solvers.SolverSP(**kwargs)
    elif solver.lower() == "smp":
        solver_object = solvers.SolverSMP(**kwargs)
    elif solver.lower() == "ssmp":
        solver_object = solvers.SolverSSMP(**kwargs)
    elif solver.lower() == "subsmp":
        solver_object = solvers.SolverSubSMP(**kwargs)
    elif solver.lower() == "bcs":
        solver_object = solvers.SolverBCS(**kwargs)
    else:
        msg = f"{solver} is not available. Please choose from one of the following: {config.all_solvers}"
        raise AttributeError(msg)
    return solver_object


