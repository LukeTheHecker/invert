import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator

class SolverMVAB(BaseSolver):
    ''' Class for the Minimum Variance Adaptive Beamformer (MVAB) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Minimum Variance Adaptive Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        self.forward = forward
        leadfield = self.forward['sol']['data']
        n_chans, n_dipoles = leadfield.shape
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        y = evoked.data
        y -= y.mean(axis=0)
        R_inv = np.linalg.inv(y@y.T)
        leadfield -= leadfield.mean(axis=0)
        
  
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = 1/(leadfield.T @ R_inv @ leadfield + alpha * np.identity(n_dipoles)) @ leadfield.T @ R_inv
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)