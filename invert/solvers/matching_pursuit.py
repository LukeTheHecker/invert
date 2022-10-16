import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator
from ..util import calc_residual_variance, thresholding, find_corner

class SolverSOMP(BaseSolver):
    ''' Class for the Simultaneous Orthogonal Matching Pursuit (S-OMP) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Simultaneous Orthogonal Matching Pursuit", **kwargs):
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        source_mat = self.calc_smop_solution(evoked.data)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_smop_solution(self, y):
        """ Calculates the S-OMP inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, time).
        
        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, time)
        """
        n_chans, n_time = y.shape
        _, n_dipoles = self.leadfield.shape

        leadfield_pinv = np.linalg.pinv(self.leadfield)
        x_hat = np.zeros((n_dipoles, n_time))
        x_hats = [deepcopy(x_hat)]
        residuals = np.array([np.linalg.norm(y - self.leadfield@x_hat), ])
        unexplained_variance = np.array([calc_residual_variance(y, self.leadfield@x_hat),])
        source_norms = np.array([0,])

        R = deepcopy(y)
        omega = np.array([])
        q = 1
        for i in range(n_chans):
            b_n = np.linalg.norm(self.leadfield.T@R, axis=1, ord=q)

            # if len(omega)>0:
            #     b_n[omega] = 0

            b_thresh = thresholding(b_n, 1)
            omega = np.append(omega, np.where(b_thresh!=0)[0])  # non-zero idc
            omega = np.unique(omega.astype(int))
            leadfield_pinv = np.linalg.pinv(self.leadfield[:, omega])
            x_hat[omega] = leadfield_pinv @ y
            R = y - self.leadfield@x_hat
            
            residuals = np.append(residuals, np.linalg.norm(R))
            unexplained_variance = np.append(unexplained_variance, calc_residual_variance(y, self.leadfield@x_hat))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append( deepcopy(x_hat) )

        unexplained_variance[0] = unexplained_variance[1]
        iters = np.arange(len(residuals))
        corner_idx = find_corner(residuals, iters)
        x_hat = x_hats[corner_idx]
        return x_hat

class SolverCOSAMP(BaseSolver):
    ''' Class for the CoSa Matching Pursuit (S-OMP) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="CoSa Matching Pursuit", **kwargs):
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        evoked.set_eeg_reference("average", projection=True, verbose=0).apply_proj()
        source_mat = np.stack([self.calc_cosamp_solution(y) for y in evoked.data.T], axis=1)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_cosamp_solution(self, y, K=25):
        """ Calculates the CoSaMP inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, time).
        K : int
            Positive integer determining the sparsity of the reconstructed signal.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, time)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        
        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        b = np.zeros((n_dipoles, ))
        r = deepcopy(y)

        residuals = np.array([np.linalg.norm(y - self.leadfield@x_hat), ])
        source_norms = np.array([0,])

        for i in range(1, n_chans+1):
            e = self.leadfield.T @ r
            e_thresh = thresholding(e, 2*K)
            omega = np.where(e_thresh!=0)[0]
            old_activations = np.where(x_hats[i-1]!=0)[0]
            T = np.unique(np.concatenate([omega, old_activations]))
            leadfield_pinv = np.linalg.pinv(self.leadfield[:, T])
            b[T] = leadfield_pinv @ y
            x_hat = thresholding(b, K)
            r = y - self.leadfield@x_hat
            
            residuals = np.append(residuals, np.linalg.norm(y - self.leadfield@x_hat))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))
            if residuals[-1] == residuals[-2]:
                break

        iters = np.arange(len(residuals))
        corner_idx = find_corner(residuals, iters)
        x_hat = x_hats[corner_idx]
        return x_hat
        