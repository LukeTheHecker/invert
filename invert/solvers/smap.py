import numpy as np
import mne
from scipy.sparse.csgraph import laplacian
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score
from scipy.sparse.csgraph import laplacian

from invert.util.util import pos_from_forward

from .base import BaseSolver, InverseOperator

class SolverSMAP(BaseSolver):
    ''' Class for the Quadratic regularization and spatial regularization
        (S-MAP) inverse solution [1].
    
    Attributes
    ----------

    References
    ----------
    [1] Baillet, S., & Garnero, L. (1997). A Bayesian approach to introducing
    anatomo-functional priors in the EEG/MEG inverse problem. IEEE transactions
    on Biomedical Engineering, 44(5), 374-385.
    
    '''
    def __init__(self, name="S-MAP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha='auto', **kwargs):
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
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        LTL = self.leadfield.T @ self.leadfield 
        # n_chans, n_dipoles = self.leadfield.shape

        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        gradient = laplacian(adjacency)
        GTG = gradient.T @ gradient
                
        inverse_operators = []
        # GG_inv = np.linalg.inv(GTG)
        for alpha in self.alphas:
            inverse_operator = np.linalg.inv(LTL + alpha * GTG) @ self.leadfield.T
            # inverse_operator = GG_inv @ self.leadfield.T @ np.linalg.inv(self.leadfield @ GG_inv @ self.leadfield.T + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        ''' Apply the S-MAP inverse operator.
        Parameters
        ----------
        evoked : mne.Evoked
            The evoke data object.

        Return
        ------
        stc : mne.SourceEstimate
            The inverse solution object.
        '''
        return super().apply_inverse_operator(evoked)
    
