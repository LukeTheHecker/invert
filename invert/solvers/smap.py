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
    '''
    def __init__(self, name="S-MAP"):
        self.name = name
        return super().__init__()

    def make_inverse_operator(self, forward, alpha='auto', verbose=0):
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
        pos = pos_from_forward(self.forward)
        n_chans, _ = leadfield.shape
        B = np.diag(np.linalg.norm(leadfield, axis=0))
        # gradient = np.gradient(B)[0] #np.gradient(B)[0]
        gradient = (np.gradient(B)[0] + np.gradient(B)[1]) / 2

        # adjacency = mne.spatial_src_adjacency(forward['src'], verbose=verbose).toarray()
        # gradient = np.gradient(adjacency)
        # gradient = np.stack([abs(gradient[0]), abs(gradient[1])], axis=0).mean(axis=0)

        # gradient = np.gradient(leadfield.T @ leadfield)
        # gradient = 1/np.stack([*gradient], axis=0).mean(axis=0)
        # gradient *= adjacency

        # adjacency = mne.spatial_src_adjacency(forward['src'], verbose=0).toarray()
        # gradient = np.gradient(adjacency)
        # gradient = np.stack([abs(gradient[0]), abs(gradient[1])], axis=0).mean(axis=0)
        # gradient = gaussian_gradient_magnitude(leadfield.T @ leadfield, sigma=1)

        # dist = cdist(pos, pos)      
        # gradient = (abs(np.gradient(dist)[0]) + abs(np.gradient(dist)[1])) / 2
        # gradient *= adjacency
        
        if isinstance(alpha, (int, float)):
            alphas = [alpha,]
        else:
            # eigenvals = np.linalg.eig(leadfield @ leadfield.T)[0]
            # alphas = [r_value * np.max(eigenvals) / 2e4 for r_value in self.r_values]
            # alphas = self.r_values
            alphas = np.insert(np.logspace(-3, 1, 12), 0, 0)
        
        inverse_operators = []
        for alpha in alphas:
            inverse_operator = np.linalg.inv(leadfield.T @ leadfield + alpha * gradient.T @ gradient) @ leadfield.T
            inverse_operators.append(inverse_operator)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)