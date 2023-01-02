from time import monotonic
import numpy as np
import mne
from .util import pos_from_forward
import matplotlib.pyplot as plt

from invert import solvers
from . import config


def Solver(solver="mne", **kwargs):
    """ Solver class ...
    """
    # Minimum Norm Algorithms
    if solver.lower() == "mne":
        solver_object = solvers.SolverMNE(**kwargs)
    elif solver.lower() == "wmne":
        solver_object = solvers.SolverWMNE(**kwargs)
    elif solver.lower() == "dspm":
        solver_object = solvers.SolverDSPM(**kwargs)
    elif solver.lower() == "l1" or solver.lower() == "fista" or solver.lower() == "mce" or solver.lower() == "minimum current estimate":
        solver_object = solvers.SolverMinimumL1Norm(**kwargs)
    elif solver.lower() == "gpt" or solver.lower() == "l1 gpt" or solver.lower() == "l1gpt":
        solver_object = solvers.SolverMinimumL1NormGPT(**kwargs)
    elif solver.lower() == "l1l2":
        solver_object = solvers.SolverMinimumL1L2Norm(**kwargs)
    
    
    # LORETA Algorithms
    elif solver.lower() == "loreta" or solver.lower() == "lor":
        solver_object = solvers.SolverLORETA(**kwargs)
    elif solver.lower() == "sloreta" or solver.lower() == "slor":
        solver_object = solvers.SolverSLORETA(**kwargs)
    elif solver.lower() == "eloreta" or solver.lower() == "elor":
        solver_object = solvers.SolverELORETA(**kwargs)
    
    # Various smooth Algorithms
    elif solver.lower() == "laura" or solver.lower() == "laur":
        solver_object = solvers.SolverLAURA(**kwargs)
    elif solver.lower() == "backus-gilbert" or solver.lower() == "b-g" or  solver.lower() == "bg":
        solver_object = solvers.SolverBackusGilbert(**kwargs)
    elif solver.lower() == "s-map" or solver.lower() == "smap":
        solver_object = solvers.SolverSMAP(**kwargs)
    
    # Bayesian Algorithms
    elif solver.lower() == "champagne" or solver.lower() == "champ":
        solver_object = solvers.SolverChampagne(**kwargs)
    elif solver.lower() == "lowsnrchampagne" or solver.lower() == "low snr champagne" or solver.lower() == "lsc":
        solver_object = solvers.SolverLowSNRChampagne(**kwargs)
    elif solver.lower() == "emchampagne" or solver.lower() == "em champagne" or solver.lower() == "emc":
        solver_object = solvers.SolverEMChampagne(**kwargs)
    elif solver.lower() == "mmchampagne" or solver.lower() == "mm champagne" or solver.lower() == "mmc":
        solver_object = solvers.SolverMMChampagne(**kwargs)
    # elif solver.lower() == "temchampagne" or solver.lower() == "tem champagne" or solver.lower() == "temc":
    #     solver_object = solvers.SolverTEMChampagne(**kwargs)
    elif solver.lower() == "gamma-map" or solver.lower() == "gamma map" or solver.lower() == "gmap":
        solver_object = solvers.SolverGammaMAP(**kwargs)
    elif solver.lower() == "source-map" or solver.lower() == "source map":
        solver_object = solvers.SolverSourceMAP(**kwargs)
    elif solver.lower() == "gamma-map-msp" or solver.lower() == "gamma map msp":
        solver_object = solvers.SolverGammaMAPMSP(**kwargs)
    elif solver.lower() == "source-map-msp" or solver.lower() == "source map msp":
        solver_object = solvers.SolverSourceMAPMSP(**kwargs)
    
    # Beamformer Algorithms
    elif solver.lower() == "mvab":
        solver_object = solvers.SolverMVAB(**kwargs)
    elif solver.lower() == "lcmv":
        solver_object = solvers.SolverLCMV(**kwargs)
    elif solver.lower() == "smv":
        solver_object = solvers.SolverSMV(**kwargs)
    elif solver.lower() == "wnmv":
        solver_object = solvers.SolverWNMV(**kwargs)
    elif solver.lower() == "hocmv":
        solver_object = solvers.SolverHOCMV(**kwargs)
    elif solver.lower() == "esmv":
        solver_object = solvers.SolverESMV(**kwargs)
    elif solver.lower() == "mcmv":
        solver_object = solvers.SolverMCMV(**kwargs)
    # elif solver.lower() == "esmcmv":
    #     solver_object = solvers.SolverESMCMV(**kwargs)
    elif solver.lower() == "recipsiicos":
        solver_object = solvers.SolverReciPSIICOS(**kwargs)
    elif solver.lower() == "sam":
        solver_object = solvers.SolverSAM(**kwargs)

    # Own approaches
    elif solver.lower() == "fully-connected" or solver.lower() == "fc" or solver.lower() == "fullyconnected" or solver.lower() == "esinet":
        solver_object = solvers.SolverFC(**kwargs)
    elif solver.lower() == "covcnn" or solver.lower() == "cov cnn" or solver.lower() == "covnet" or solver.lower() == "cov-cnn":
        solver_object = solvers.SolverCovCNN(**kwargs)
    elif solver.lower() == "lstm":
        solver_object = solvers.SolverLSTM(**kwargs)
    elif solver.lower() == "cnn":
        solver_object = solvers.SolverCNN(**kwargs)
    
    # Matching Pursuit/ Compressive Sensing
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
    elif solver.lower() == "ssp":
        solver_object = solvers.SolverSSP(**kwargs)
    elif solver.lower() == "smp":
        solver_object = solvers.SolverSMP(**kwargs)
    elif solver.lower() == "ssmp":
        solver_object = solvers.SolverSSMP(**kwargs)
    elif solver.lower() == "subsmp":
        solver_object = solvers.SolverSubSMP(**kwargs)
    elif solver.lower() == "isubsmp":
        solver_object = solvers.SolverISubSMP(**kwargs)
    elif solver.lower() == "bcs":
        solver_object = solvers.SolverBCS(**kwargs)
    
    # Subspace Methods
    elif solver.lower() == "music":
        solver_object = solvers.SolverMUSIC(**kwargs)
    elif solver.lower() == "rap-music" or solver.lower() == "rap music" or solver.lower() == "rap":
        solver_object = solvers.SolverRAPMUSIC(**kwargs)
    elif solver.lower() == "trap-music" or solver.lower() == "trap music" or solver.lower() == "trap":
        solver_object = solvers.SolverTRAPMUSIC(**kwargs)
    elif solver.lower() == "flex-music" or solver.lower() == "flex music" or solver.lower() == "flex":
        solver_object = solvers.SolverFLEXMUSIC(**kwargs)
    
    # Other
    elif solver.lower() == "epifocus":
        solver_object = solvers.SolverEPIFOCUS(**kwargs)

    
    
    
    else:
        msg = f"{solver} is not available. Please choose from one of the following: {config.all_solvers}"
        raise AttributeError(msg)
    return solver_object


