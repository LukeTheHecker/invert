from copy import deepcopy
from scipy.spatial.distance import cdist
from scipy.sparse import spdiags
from scipy.linalg import inv
import numpy as np
import mne
from scipy.fftpack import dct
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from ..util import pos_from_forward
# from ..invert import BaseSolver, InverseOperator
# from .. import invert
from .base import BaseSolver, InverseOperator

# from .. import invert
# import BaseSolver, InverseOperator


class SolverChampagne(BaseSolver):
    ''' Class for the Champagne inverse solution. Code is based on the
    implementation from the BSI-Zoo: https://github.com/braindatalab/BSI-Zoo/
    
    References
    ----------
    [1] Owen, J., Attias, H., Sekihara, K., Nagarajan, S., & Wipf, D. (2008).
    Estimating the location and orientation of complex, correlated neural
    activity using MEG. Advances in Neural Information Processing Systems, 21.
    
    [2] Wipf, D. P., Owen, J. P., Attias, H. T., Sekihara, K., & Nagarajan, S.
    S. (2010). Robust Bayesian estimation of the location, orientation, and time
    course of multiple correlated neural sources using MEG. NeuroImage, 49(1),
    641-655. 
    
    [3] Owen, J. P., Wipf, D. P., Attias, H. T., Sekihara, K., &
    Nagarajan, S. S. (2012). Performance evaluation of the Champagne source
    reconstruction algorithm on simulated and real M/EEG data. Neuroimage,
    60(1), 305-323.
    '''

    def __init__(self, name="Champagne", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', max_iter=1000, noise_cov=None, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum number of iterations.
        noise_cov : [None, numpy.ndarray]
            The noise covariance matrix. Use "None" if not available.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        self.forward = forward
        self.leadfield = self.forward['sol']['data']
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.champagne(evoked.data, alpha, max_iter=max_iter,)
            inverse_operators.append( inverse_operator )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:

        return super().apply_inverse_operator(evoked)
    
    
    def champagne(self, y, alpha, max_iter=1000):
        ''' Champagne method.

        Parameters
        ----------
        y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.
        max_iter : int, optional
            The maximum number of inner loop iterations

        Returns
        -------
        x : array, shape (dipoles, time)
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        
        '''
        _, n_sources = self.leadfield.shape
        _, n_times = y.shape
        leadfield = deepcopy(self.leadfield)
        gammas = np.ones(n_sources)
        eps = np.finfo(float).eps
        threshold = 0.2 * np.mean(np.diag(self.noise_cov))
        # x = np.zeros((n_sources, n_times))
        n_active = n_sources
        active_set = np.arange(n_sources)
        # H = np.concatenate(L, np.eyes(n_sensors), axis = 1)
        self.noise_cov = alpha*self.noise_cov
        x_bars = []
        for i in range(max_iter):
            gammas[np.isnan(gammas)] = 0.0
            gidx = np.abs(gammas) > threshold
            active_set = active_set[gidx]
            gammas = gammas[gidx]

            # update only active gammas (once set to zero it stays at zero)
            if n_active > len(active_set):
                n_active = active_set.size
                leadfield = leadfield[:, gidx]

            Gamma = spdiags(gammas, 0, len(active_set), len(active_set))
            # Calculate Source Covariance Matrix based on currently selected gammas
            Sigma_y = (leadfield @ Gamma @ leadfield.T) + self.noise_cov
            U, S, _ = np.linalg.svd(Sigma_y, full_matrices=False)
            S = S[np.newaxis, :]
            del Sigma_y
            Sigma_y_inv = np.dot(U / (S + eps), U.T)
            # Sigma_y_inv = linalg.inv(Sigma_y)
            x_bar = Gamma @ leadfield.T @ Sigma_y_inv @ y

            # old gamma calculation throws warning
            # gammas = np.sqrt(
            #     np.diag(x_bar @ x_bar.T / n_times) / np.diag(leadfield.T @ Sigma_y_inv @ leadfield)
            # )
            # Calculate gammas 
            gammas = np.diag(x_bar @ x_bar.T / n_times) / np.diag(leadfield.T @ Sigma_y_inv @ leadfield)
            # set negative gammas to nan to avoid bad sqrt
            gammas.astype(np.float64)  # this is required for numpy to accept nan
            gammas[gammas<0] = np.nan
            gammas = np.sqrt(gammas)

            # Calculate Residual to the data
            e_bar = y - (leadfield @ x_bar)
            self.noise_cov = np.sqrt(np.diag(e_bar @ e_bar.T / n_times) / np.diag(Sigma_y_inv))
            threshold = 0.2 * np.mean(np.diag(self.noise_cov))
            x_bars.append(x_bar)

            if i>0 and np.linalg.norm(x_bars[-1]) == 0:
                x_bar = x_bars[-2]
                break
        # active_set
        gammas_full = np.zeros(n_sources)
        gammas_full[active_set] = gammas
        Gamma_full = spdiags(gammas_full, 0, n_sources, n_sources)
        Sigma_y = (self.leadfield @ Gamma_full @ self.leadfield.T) + self.noise_cov
        U, S, _ = np.linalg.svd(Sigma_y, full_matrices=False)
        S = S[np.newaxis, :]
        del Sigma_y
        Sigma_y_inv_full = np.dot(U / (S + eps), U.T)
        inverse_operator = Gamma_full @ self.leadfield.T @ Sigma_y_inv_full

        return inverse_operator

# class SolverChampagne(BaseSolver):
#     ''' Class for the Champagne inverse solution. Code is based on the
#     implementation from the BSI-Zoo: https://github.com/braindatalab/BSI-Zoo/
    
#     References
#     ----------
#     [1] Owen, J., Attias, H., Sekihara, K., Nagarajan, S., & Wipf, D. (2008).
#     Estimating the location and orientation of complex, correlated neural
#     activity using MEG. Advances in Neural Information Processing Systems, 21.
    
#     [2] Wipf, D. P., Owen, J. P., Attias, H. T., Sekihara, K., & Nagarajan, S.
#     S. (2010). Robust Bayesian estimation of the location, orientation, and time
#     course of multiple correlated neural sources using MEG. NeuroImage, 49(1),
#     641-655. 
    
#     [3] Owen, J. P., Wipf, D. P., Attias, H. T., Sekihara, K., &
#     Nagarajan, S. S. (2012). Performance evaluation of the Champagne source
#     reconstruction algorithm on simulated and real M/EEG data. Neuroimage,
#     60(1), 305-323.
#     '''

#     def __init__(self, name="Champagne", **kwargs):
#         self.name = name
#         return super().__init__(**kwargs)

#     def make_inverse_operator(self, forward, *args, alpha='auto', max_iter=1000, noise_cov=None, verbose=0, **kwargs):
#         ''' Calculate inverse operator.

#         Parameters
#         ----------
#         forward : mne.Forward
#             The mne-python Forward model instance.
#         alpha : float
#             The regularization parameter.
        
#         Return
#         ------
#         self : object returns itself for convenience
#         '''
#         self.forward = forward
#         self.leadfield = self.forward['sol']['data']
#         super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
#         n_chans = self.leadfield.shape[0]
#         if noise_cov is None:
#             noise_cov = np.identity(n_chans)

#         self.noise_cov = noise_cov
#         self.inverse_operators = []
#         return self

#     def apply_inverse_operator(self, evoked, max_iter=1000) -> mne.SourceEstimate:

#         source_mat = self.champagne(evoked.data, max_iter=max_iter)
#         stc = self.source_to_object(source_mat, evoked)
#         return stc
    
    
#     def champagne(self, y, max_iter=1000):
#         """Champagne method based on our MATLAB codes  
#         -> copied as mentioned in class docstring

#         Parameters
#         ----------
#         y : array, shape (n_sensors,)
#             measurement vector, capturing sensor measurements
#         max_iter : int, optional
#             The maximum number of inner loop iterations

#         Returns
#         -------
#         x : array, shape (n_sources,)
#             Parameter vector, e.g., source vector in the context of BSI (x in the cost
#             function formula).
        
#         """
#         _, n_sources = self.leadfield.shape
#         _, n_times = y.shape
#         if self.alpha == "auto":
#             self.alpha = 1
#         gammas = np.ones(n_sources)
#         eps = np.finfo(float).eps
#         threshold = 0.2 * np.mean(np.diag(self.noise_cov))
#         x = np.zeros((n_sources, n_times))
#         n_active = n_sources
#         active_set = np.arange(n_sources)
#         # H = np.concatenate(L, np.eyes(n_sensors), axis = 1)
#         self.noise_cov = self.alpha*self.noise_cov
#         x_bars = []
#         for i in range(max_iter):
#             gammas[np.isnan(gammas)] = 0.0
#             gidx = np.abs(gammas) > threshold
#             active_set = active_set[gidx]
#             gammas = gammas[gidx]

#             # update only active gammas (once set to zero it stays at zero)
#             if n_active > len(active_set):
#                 n_active = active_set.size
#                 self.leadfield = self.leadfield[:, gidx]

#             Gamma = spdiags(gammas, 0, len(active_set), len(active_set))
#             # Calculate Source Covariance Matrix based on currently selected gammas
#             Sigma_y = (self.leadfield @ Gamma @ self.leadfield.T) + self.noise_cov
#             U, S, _ = np.linalg.svd(Sigma_y, full_matrices=False)
#             S = S[np.newaxis, :]
#             del Sigma_y
#             Sigma_y_inv = np.dot(U / (S + eps), U.T)
#             # Sigma_y_inv = linalg.inv(Sigma_y)
#             x_bar = Gamma @ self.leadfield.T @ Sigma_y_inv @ y

#             gammas = np.sqrt(
#                 np.diag(x_bar @ x_bar.T / n_times) / np.diag(self.leadfield.T @ Sigma_y_inv @ self.leadfield)
#             )
#             # Calculate Residual to the data
#             e_bar = y - (self.leadfield @ x_bar)
#             self.noise_cov = np.sqrt(np.diag(e_bar @ e_bar.T / n_times) / np.diag(Sigma_y_inv))
#             threshold = 0.2 * np.mean(np.diag(self.noise_cov))
#             x_bars.append(x_bar)

#             if i>0 and np.linalg.norm(x_bars[-1]) == 0:
#                 x_bar = x_bars[-2]
#                 break

#         x[active_set, :] = x_bar

#         return x

