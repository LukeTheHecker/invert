from time import monotonic
import numpy as np
import mne
from .util import pos_from_forward
import esinet
import matplotlib.pyplot as plt

from invert import solvers
from . import config


def Solver(solver="mne"):
    """ Solver class ...
    """

    if solver.lower() == "mne":
        solver_object = solvers.SolverMinimumNorm()
    elif solver.lower() == "wmne":
        solver_object = solvers.SolverWeightedMinimumNorm()
    elif solver.lower() == "dspm":
        solver_object = solvers.SolverDynamicStatisticalParametricMapping()
    elif solver.lower() == "loreta" or solver.lower() == "lor":
        solver_object = solvers.SolverLORETA()
    elif solver.lower() == "sloreta" or solver.lower() == "slor":
        solver_object = solvers.SolverSLORETA()
    elif solver.lower() == "eloreta" or solver.lower() == "elor":
        solver_object = solvers.SolverELORETA()
    elif solver.lower() == "laura" or solver.lower() == "laur":
        solver_object = solvers.SolverLAURA()
    elif solver.lower() == "backus-gilbert" or solver.lower() == "b-g" or  solver.lower() == "bg":
        solver_object = solvers.SolverBackusGilbert()
    elif solver.lower() == "s-map" or solver.lower() == "smap":
        solver_object = solvers.SolverSMAP()
    elif solver.lower() == "multiple sparse priors" or solver.lower() == "msp":
        inversion_type = "MSP"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type)
    elif solver.lower() == "bayesian loreta" or solver.lower() == "bayesian lor" or solver.lower() == "bloreta" or solver.lower() == "blor":
        inversion_type = "LORETA"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type)
    elif solver.lower() == "bayesian mne" or solver.lower() == "bmne":
        inversion_type = "MNE"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type)
    elif solver.lower() == "bayesian beamformer" or solver.lower() == "bbmf":
        inversion_type = "BMF"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type)
    elif solver.lower() == "bayesian beamformer loreta" or solver.lower() == "bbmf-lor" or solver.lower() == "bmf-lor":
        inversion_type = "BMF-LOR"
        solver_object = solvers.SolverMultipleSparsePriors(inversion_type=inversion_type)
    elif solver.lower() == "fully-connected" or solver.lower() == "fc" or solver.lower() == "fullyconnected" or solver.lower() == "esinet":
        solver_object = solvers.SolverFullyConnected()
    elif solver.lower() == "lucas":
        solver_object = solvers.SolverLUCAS()
    elif solver.lower() == "champagne" or solver.lower() == "champ":
        solver_object = solvers.SolverChampagne()
    # elif solver.lower() == "lstm":
    #     solver_object = make_lstm_inverse_operator(forward, kwargs["evoked"].info, verbose=verbose)
    else:
        msg = f"{solver} is not available. Please choose from one of the following: {config.all_solvers}"
        raise AttributeError(msg)
    return solver_object


