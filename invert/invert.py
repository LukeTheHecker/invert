import numpy as np
import mne
from .util import pos_from_forward
import esinet
import matplotlib.pyplot as plt
# from loreta import make_loreta_inverse_operator
# from .solvers.minimum_norm_estimates import (make_mne_inverse_operator, 
#                                     make_wmne_inverse_operator,
#                                     make_dspm_inverse_operator)
# from .solvers.loreta import (make_loreta_inverse_operator,
#                     make_sloreta_inverse_operator,
#                     make_eloreta_inverse_operator)

# from .solvers.wrop import (make_laura_inverse_operator,
#                     make_backus_gilbert_inverse_operator)

# from .solvers.multiple_sparse_priors import (make_msp_inverse_operator)

# from .solvers.esinet import (make_fullyconnected_inverse_operator,
#                             make_lstm_inverse_operator)

all_solvers = [ "MNE", "wMNE", "dSPM", 
                "LORETA", "sLORETA", "eLORETA", 
                "LAURA", "Backus-Gilbert", 
                "Multiple Sparse Priors", "Bayesian LORETA", "Bayesian MNE", "Bayesian Beamformer", "Bayesian Beamformer LORETA",
                "Fully-Connected", "LSTM"]


class InverseOperator:
    ''' This class holds the inverse operator, which may be a simple
    numpy.ndarray matrix or some object like an esinet.net()
    '''
    def __init__(self, inverse_operator, solver_name):
        self.solver_name = solver_name
        self.data = inverse_operator
        self.handle_inverse_operator()
        self.has_multiple_operators()

    def has_multiple_operators(self):
        ''' Check if there are multiple inverse_operators.'''
        if type(self.data) == list:
            if len(self.data) > 1:
                return True
        return False

    def handle_inverse_operator(self):
        if type(self.data) != list:
            self.data = [self.data,]
        self.type = type(self.data[0])
        
        

class BaseSolver:
    def __init__(self, verbose=0):
        self.verbose = verbose
        
    def make_inverse_operator(self, forward: mne.Forward, *args):
        self.forward = forward
        pass

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        ''' Apply the inverse operator '''
        # assert type(self.inverse_operator) == type(InverseOperator), f"inverse operator object is not correctly set (is of type {type(self.inverse_operator)}, must be {InverseOperator})"
        M = evoked.data

        if len(self.inverse_operator.data) == 1:
            inverse_operator = self.inverse_operator.data[0]
            source_mat = inverse_operator.data @ M 
        else:
            source_mats = []
            l2_norms = []
            inverse_operators = self.inverse_operator.data
            for inverse_operator in inverse_operators:
                source_mats.append( inverse_operator.data @ M  )
                l2_norms.append( np.linalg.norm(source_mats[-1]) )
            # corner_idx = self.find_corner(l2_norms)
            corner_idx = 5
            source_mat = source_mats[corner_idx]

            plt.figure()
            plt.plot(np.arange(12), l2_norms)
        
        stc = self.source_to_object(source_mat, evoked)
        return stc

    
    def source_to_object(self, source_mat, evoked):
        ''' Converts the source_mat matrix to an mne.SourceEstimate object '''
        # Convert source to mne.SourceEstimate object
        source_model = self.forward['src']
        vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
        tmin = evoked.tmin
        sfreq = evoked.info["sfreq"]
        tstep = 1/sfreq
        subject = evoked.info["subject_info"]

        if subject is None:
            subject = "fsaverage"
        
        stc = mne.SourceEstimate(source_mat, vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=self.verbose)
        return stc
        
    @staticmethod
    def find_corner(l2_norms):
        
        pass

# def make_inverse_operator(forward: mne.Forward, solver='MNE', alpha=0.001, 
#     noise_cov=None, source_cov=None, stop_crit=0.0005, 
#     verbose=0, drop_off=2, **kwargs) -> InverseOperator:

#     leadfield = forward['sol']['data']
#     pos = pos_from_forward(forward, verbose=verbose)
#     adjacency = mne.spatial_src_adjacency(forward['src'], verbose=verbose).toarray()
        
#     if solver.lower() == "mne":
#         inverse_operator = make_mne_inverse_operator(leadfield, alpha=alpha, noise_cov=noise_cov)
    
#     elif solver.lower() == "wmne":
#         inverse_operator = make_wmne_inverse_operator(leadfield, alpha=alpha, noise_cov=noise_cov)
    
#     elif solver.lower() == "dspm":
#         inverse_operator = make_dspm_inverse_operator(leadfield, alpha=alpha, noise_cov=noise_cov, source_cov=source_cov)
    
#     elif solver.lower() == "loreta" or solver.lower() == "lor":
#         inverse_operator = make_loreta_inverse_operator(leadfield, adjacency, alpha=alpha)
    
#     elif solver.lower() == "sloreta" or solver.lower() == "slor":
#         inverse_operator = make_sloreta_inverse_operator(leadfield, alpha=alpha, noise_cov=noise_cov)
    
#     elif solver.lower() == "eloreta" or solver.lower() == "elor":
#         inverse_operator = make_eloreta_inverse_operator(leadfield, alpha=alpha, noise_cov=noise_cov, stop_crit=stop_crit, verbose=verbose)
    
#     elif solver.lower() == "laura" or solver.lower() == "laur":
#         inverse_operator = make_laura_inverse_operator(leadfield, pos, adjacency, alpha=alpha, noise_cov=noise_cov, drop_off=drop_off)

#     elif solver.lower() == "backus-gilbert" or solver.lower() == "b-g" or  solver.lower() == "bg":
#         inverse_operator = make_backus_gilbert_inverse_operator(leadfield, pos)

#     elif solver.lower() == "multiple sparse priors" or solver.lower() == "msp":
#         if not "evoked" in kwargs:
#             msg = f"""Multiple Sparse Priors requires an evoked object: make_inverse_operator(forward, solver="Multiple Sparse Priors", evoked=evoked) """
#             raise AttributeError(msg)
#         inversion_type = "MSP"
#         inverse_operator = make_msp_inverse_operator(leadfield, pos, adjacency, inversion_type=inversion_type, **kwargs)
    
#     elif solver.lower() == "bayesian loreta" or solver.lower() == "bayesian lor" or solver.lower() == "bloreta" or solver.lower() == "blor":
#         if not "evoked" in kwargs:
#             msg = f"""Bayesian LORETA requires an evoked object: make_inverse_operator(forward, solver="Bayesian LORETA", evoked=evoked) """
#             raise AttributeError(msg)
#         inversion_type = "LORETA"
#         inverse_operator = make_msp_inverse_operator(leadfield, pos, adjacency, inversion_type=inversion_type, **kwargs)
    
#     elif solver.lower() == "bayesian mne" or solver.lower() == "bmne":
#         if not "evoked" in kwargs:
#             msg = f"""Bayesian MNE requires an evoked object: make_inverse_operator(forward, solver="Bayesian MNE", evoked=evoked) """
#             raise AttributeError(msg)
#         inversion_type = "MNE"
#         inverse_operator = make_msp_inverse_operator(leadfield, pos, adjacency, inversion_type=inversion_type, **kwargs)

#     elif solver.lower() == "bayesian beamformer" or solver.lower() == "bbmf":
#         if not "evoked" in kwargs:
#             msg = f"""Bayesian Beamformer requires an evoked object: make_inverse_operator(forward, solver="Bayesian Beamformer", evoked=evoked) """
#             raise AttributeError(msg)
#         inversion_type = "BMF"
#         inverse_operator = make_msp_inverse_operator(leadfield, pos, adjacency, inversion_type=inversion_type, **kwargs)

#     elif solver.lower() == "bayesian beamformer loreta" or solver.lower() == "bbmf-lor":
#         if not "evoked" in kwargs:
#             msg = f"""Bayesian Beamformer LORETA requires an evoked object: make_inverse_operator(forward, solver="Bayesian Beamformer LORETA", evoked=evoked) """
#             raise AttributeError(msg)
#         inversion_type = "BMF-LOR"
#         inverse_operator = make_msp_inverse_operator(leadfield, pos, adjacency, inversion_type=inversion_type, **kwargs)    
    
#     elif solver.lower() == "fully-connected" or solver.lower() == "fc" or solver.lower() == "fullyconnected" or solver.lower() == "esinet":
#         if not "evoked" in kwargs:
#             msg = f"""Fully-Connected requires an evoked object: make_inverse_operator(forward, solver="Fully-Connected", evoked=evoked) """
#             raise AttributeError(msg)

#         inverse_operator = make_fullyconnected_inverse_operator(forward, kwargs["evoked"].info, verbose=verbose)

#     elif solver.lower() == "lstm":
#         if not "evoked" in kwargs:
#             msg = f"""LSTM requires an evoked object: make_inverse_operator(forward, solver="LSTM", evoked=evoked) """
#             raise AttributeError(msg)

#         inverse_operator = make_lstm_inverse_operator(forward, kwargs["evoked"].info, verbose=verbose)


#     else:
#         msg = f"{solver} is not available. Please choose from one of the following: {all_solvers}"
#         raise AttributeError(msg)

#     inverse_operator_object = InverseOperator(inverse_operator, solver)

#     return inverse_operator_object

# def apply_inverse_operator(evoked, inverse_operator, forward, verbose=0):
#     ''' Apply the inverse operator to the evoked object to calculate the source.
    
#     Parameters
#     ----------
#     evoked : mne.EvokedArray
#         The evoked object containing evoked M/EEG data.
#     inverse_operator : invert.InverseOperator
#         The inverse operator object.
    
#     Return
#     ------
#     stc : mne.SourceEstimate
#         The source estimate object containing the source
#     '''
#     # Do some preprocessing/ whitening?
#     M = evoked.data

#     if inverse_operator.type == np.ndarray:
#         source_mat = inverse_operator.data @ M 
#     elif inverse_operator.type == list:
#         maximum_a_posteriori, A, S = inverse_operator.data
#         # transform data M with spatial (A) and temporal (S) projector
#         M_ = A @ M @ S
#         # invert transformed data M_ to tansformed sources J_
#         J_ = maximum_a_posteriori @ M_
#         # Project tansformed sources J_ back to original time frame using temporal projector S
#         source_mat =  J_ @ S.T 
#     elif inverse_operator.type == esinet.Net:
#         source_mat = inverse_operator.data.predict(evoked)[0].data
#     else:
#         msg = f"type of inverse operator ({inverse_operator.type}) unknown"
#         raise AttributeError(msg)
    
#     # Convert source to mne.SourceEstimate object
#     source_model = forward['src']
#     vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
#     tmin = evoked.tmin
#     sfreq = evoked.info["sfreq"]
#     tstep = 1/sfreq
#     subject = evoked.info["subject_info"]

#     if subject is None:
#         subject = "fsaverage"
    
#     stc = mne.SourceEstimate(source_mat, vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=verbose)
#     return stc



# from invert import make_inverse_operator

# inverse_operator = make_inverse_operator(fwd, solver="backus-gilbert", 
#         alpha='auto', noise_cov=None)
    
# stc = apply_inverse_operator(evoked, inverse_operator)

# inverse_operator = make_inverse_operator(fwd, solver="msp", 
#         alpha='auto', noise_cov=None, evoked=evoked)
    
# stc = apply_inverse_operator(evoked, inverse_operator)
