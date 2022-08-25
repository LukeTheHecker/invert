import numpy as np
import mne
from scipy.sparse.csgraph import laplacian
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score

from invert.util.util import pos_from_forward

from .base import BaseSolver, InverseOperator

class SolverSMAP(BaseSolver):
    ''' Class for the Quadratic regularization and spatial regularization
    (S-MAP) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.

    References
    ----------
    Baillet, S., & Garnero, L. (1997). A Bayesian approach to introducing
    anatomo-functional priors in the EEG/MEG inverse problem. IEEE transactions
    on Biomedical Engineering, 44(5), 374-385.
    
    '''
    def __init__(self, name="S-MAP"):
        self.name = name
        return super().__init__()

    def make_inverse_operator(self, forward, *args, alpha='auto', verbose=0):
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
        gradient = self.calculate_gradient(verbose=verbose)
       
        if isinstance(alpha, (int, float)):
            alphas = [alpha,]
        else:
            eigenvals = np.linalg.eig(leadfield @ leadfield.T)[0]
            alphas = [r_value * np.max(eigenvals) / 2e4 for r_value in self.r_values]
            # alphas = self.r_values
            # alphas = np.insert(np.logspace(-3, 1, 12), 0, 0)
        
        inverse_operators = []
        # GG_inv = np.linalg.inv(gradient.T @ gradient)
        for alpha in alphas:
            inverse_operator = np.linalg.inv(leadfield.T @ leadfield + alpha * gradient.T @ gradient) @ leadfield.T
            # inverse_operator = GG_inv @ leadfield.T @ np.linalg.inv(leadfield @ GG_inv @ leadfield.T + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)
    
    def calculate_gradient(self, verbose=0):
        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=verbose).toarray()

        gradient = adjacency
        n_dipoles = gradient.shape[0]
        for i in range(n_dipoles):
            row = gradient[i,:]
            gradient[i,i] = np.sum(row)-1
            gradient[i, row==1] = -1
        return gradient