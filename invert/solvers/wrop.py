import numpy as np
from scipy.spatial.distance import cdist
import mne
# from ..invert import BaseSolver, InverseOperator
from .base import BaseSolver, InverseOperator
from ..util import pos_from_forward

class SolverBackusGilbert(BaseSolver):
    ''' Class for the Backus Gilbert inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Backus-Gilbert", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha='auto', verbose=0, **kwargs):
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
        _, n_dipoles = self.leadfield.shape
        pos = pos_from_forward(forward, verbose=verbose)
        dist = cdist(pos, pos)
        if verbose>0:
            print(f"No regularization possible with {self.name} - alpha value is not used")
        

        W_BG = []
        for i in range(n_dipoles):
            W_gamma_BG = np.diag(dist[i, :])
            W_BG.append(W_gamma_BG)

        C = []
        for i in range(n_dipoles):
            C_gamma = self.leadfield @ W_BG[i] @ self.leadfield.T
            C.append(C_gamma)

        F = self.leadfield @ self.leadfield.T

        E = []
        for i in range(n_dipoles):
            E_gamma = C[i] + F
            E.append(E_gamma)

        L = self.leadfield @ np.ones((n_dipoles, 1))

        T = []
        for i in range(n_dipoles):
            E_gamma_pinv = np.linalg.pinv(E[i])
            T_gamma = (E_gamma_pinv @ L) / (L.T @ E_gamma_pinv @ L)
            T.append(T_gamma)

        inverse_operators = [np.stack(T, axis=0)[:, :, 0],]
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

class SolverLAURA(BaseSolver):
    ''' Class for the Local AUtoRegressive Average (LAURA) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Local Auto-Regressive Average", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, noise_cov=None, alpha='auto', drop_off=2, verbose=0, **kwargs):
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
        adjacency = mne.spatial_src_adjacency(forward['src'], verbose=verbose).toarray()
        n_chans, _ = self.leadfield.shape
        pos = pos_from_forward(forward, verbose=verbose)
        if noise_cov is None:
            noise_cov = np.identity(n_chans)

        d = cdist(pos, pos)

        # Get the adjacency matrix of the source spaces
        for i in range(d.shape[0]):
            # find dipoles that are no neighbor to dipole i
            non_neighbors = np.where(~adjacency.astype(bool)[i, :])[0]
            # append dipole itself
            non_neighbors = np.append(non_neighbors, i)
            # set non-neighbors to zero
            d[i, non_neighbors] = 0
        A = -d
        A[A!=0] **= -drop_off
        A[np.isinf(A)] = 0
        W = np.identity(A.shape[0])
        M_j = W @ A

        # Source Space metric
        W_j = np.linalg.inv(M_j.T @ M_j)
        W_j_inv = np.linalg.inv(W_j)
        W_d = np.linalg.inv(noise_cov)

       
 
        inverse_operators = []
        for alpha in self.alphas:
            noise_term = (alpha**2) * np.linalg.inv(W_d)
            inverse_operator = W_j_inv @ self.leadfield.T @ np.linalg.inv(self.leadfield @ W_j_inv @ self.leadfield.T + noise_term)
            inverse_operators.append(inverse_operator)
            
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)