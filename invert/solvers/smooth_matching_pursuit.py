import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator
from ..util import calc_residual_variance, thresholding, find_corner

class SolverSMP(BaseSolver):
    ''' Class for the Smooth Matching Pursuit (SMP) inverse
        solution. Developed by Lukas Hecker, 19.10.2022
    
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    '''
    def __init__(self, name="Smooth Matching Pursuit", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

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
        self.leadfield = leadfield
        
        
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, K=1) -> mne.SourceEstimate:
        source_mat = np.stack([self.calc_omp_solution(y, K=K) for y in evoked.data.T], axis=1)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_omp_solution(self, y, K=1):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        
        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        # leadfield_pinv = np.linalg.pinv(self.leadfield)
        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        # unexplained_variance = np.array([calc_residual_variance(y, leadfield@x_hat),])
        source_norms = np.array([0,])

        x_hat = np.zeros((n_dipoles, ))
        omega = np.array([])
        r = deepcopy(y)
        residuals = np.array([np.linalg.norm(y - self.leadfield@x_hat), ])
        source_norms = np.array([0,])
        x_hats = [deepcopy(x_hat), ]

        for i in range(n_chans):
            b = self.leadfield.T @ r
            b_thresh = thresholding(b, K)
            omega = np.append(omega, np.where(b_thresh!=0)[0])  # non-zero idc
            omega = omega.astype(int)

            x_hat[omega] = np.linalg.pinv(self.leadfield[:, omega]) @ y
            r = y - self.leadfield@x_hat

            residuals = np.append(residuals, np.linalg.norm(y - self.leadfield@x_hat))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))


            
        iters = np.arange(len(residuals)).astype(float)
        corner_idx = find_corner(iters, residuals)
        x_hat = x_hats[corner_idx]
        return x_hat
