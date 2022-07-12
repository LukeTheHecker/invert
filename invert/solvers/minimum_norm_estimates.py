import numpy as np
import mne
from .base import BaseSolver, InverseOperator

class SolverMinimumNorm(BaseSolver):
    ''' Class for the Minimum Norm Estimate (MNE) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Minimum Norm Estimate"):
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
        n_chans, _ = leadfield.shape
        
        if isinstance(alpha, (int, float)):
            alphas = [alpha,]
        else:
            eigenvals = np.linalg.eig(leadfield @ leadfield.T)[0]
            alphas = [r_value * np.max(eigenvals) / 2e4 for r_value in self.r_values]
        
        inverse_operators = []
        for alpha in alphas:
            inverse_operator = leadfield.T @ np.linalg.inv(leadfield @ leadfield.T + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)


class SolverWeightedMinimumNorm(BaseSolver):
    ''' Class for the Weighted Minimum Norm Estimate (wMNE) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Weighted Minimum Norm Estimate"):
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
        W = np.diag(np.linalg.norm(leadfield, axis=0))

        n_chans, _ = leadfield.shape

        if isinstance(alpha, (int, float)):
            alphas = [alpha,]
        else:
            eigenvals = np.linalg.eig(leadfield @ W @ leadfield.T)[0]
            alphas = [r_value * np.max(eigenvals) / 2e4 for r_value in self.r_values]

        inverse_operators = []
        for alpha in alphas:
            inverse_operator = np.linalg.inv(W.T @ W) @ leadfield.T  @ np.linalg.inv(leadfield @ np.linalg.inv(W.T @ W) @ leadfield.T + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)


class SolverDynamicStatisticalParametricMapping(BaseSolver):
    ''' Class for the Dynamic Statistical Parametric Mapping (dSPM) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Dynamic Statistical Parametric Mapping"):
        self.name = name
        return super().__init__()

    def make_inverse_operator(self, forward, alpha=0.01, noise_cov=None, source_cov=None,
                            verbose=0):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        noise_cov : numpy.ndarray
            The noise covariance matrix (channels x channels).
        source_cov : numpy.ndarray
            The source covariance matrix (dipoles x dipoles). This can be used if
            prior information, e.g., from fMRI images, is available.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        self.forward = forward
        leadfield = self.forward['sol']['data']
        n_chans, n_dipoles = leadfield.shape

        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        if source_cov is None:
            source_cov = np.identity(n_dipoles)
        

        if isinstance(alpha, (int, float)):
            alphas = [alpha,]
        else:
            # eigenvals = np.linalg.eig(leadfield @ source_cov @ leadfield.T)[0]
            # alphas = [r_value * np.max(eigenvals) / 2e4 for r_value in self.r_values]
            # alphas = self.r_values
            # alphas = self.r_values = np.insert(np.logspace(-6, 6, 50), 0, 0)
            print(f"alpha must be set to a float when using {self.name}, auto does not work yet.")
            alphas = [0.01,]
        inverse_operators = []
        leadfield_source_cov = source_cov @ leadfield.T
        
        for alpha in alphas:
            K = leadfield_source_cov @ np.linalg.inv(leadfield @ leadfield_source_cov + alpha * noise_cov)
            W_dSPM = np.diag( np.sqrt( 1 / np.diagonal(K @ noise_cov @ K.T) ) )
            inverse_operator = W_dSPM @ K
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)