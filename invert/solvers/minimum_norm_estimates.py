import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator

class SolverMinimumNorm(BaseSolver):
    ''' Class for the Minimum Norm Estimate (MNE) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Minimum Norm Estimate", **kwargs):
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
        self.alphas = alphas
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
    def __init__(self, name="Weighted Minimum Norm Estimate", **kwargs):
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
        W = np.diag(np.linalg.norm(leadfield, axis=0))
        WTW = np.linalg.inv(W.T @ W)
        LWTWL = leadfield @ WTW @ leadfield.T
        n_chans, _ = leadfield.shape

        if isinstance(alpha, (int, float)):
            alphas = [alpha,]
        else:
            eigenvals = np.linalg.eig(leadfield @ W @ leadfield.T)[0]
            alphas = [r_value * np.max(eigenvals) / 2e4 for r_value in self.r_values]

        inverse_operators = []
        for alpha in alphas:
            inverse_operator = WTW @ leadfield.T  @ np.linalg.inv(LWTWL + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        self.alphas = alphas
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
    def __init__(self, name="Dynamic Statistical Parametric Mapping", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha=0.01, noise_cov=None, source_cov=None,
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
            alphas = self.r_values
            # alphas = self.r_values = np.insert(np.logspace(-6, 6, 50), 0, 0)
            # print(f"alpha must be set to a float when using {self.name}, auto does not work yet.")
            # alphas = [0.01,]
        inverse_operators = []
        leadfield_source_cov = source_cov @ leadfield.T
        LLS = leadfield @ leadfield_source_cov
        for alpha in alphas:
            K = leadfield_source_cov @ np.linalg.inv(LLS + alpha * noise_cov)
            W_dSPM = np.diag( np.sqrt( 1 / np.diagonal(K @ noise_cov @ K.T) ) )
            inverse_operator = W_dSPM @ K
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        self.alphas = alphas
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)
    

class SolverMinimumL1Norm(BaseSolver):
    ''' Class for the Minimum Current Estimate (MCE) inverse solution using the FISTA solver.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Minimum Current Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha='auto', max_iter=1000, noise_cov=None, verbose=0):
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
        if self.verbose>0:
            print(f"No inverse operator is computed for {self.name}")
        self.forward = forward
        self.leadfield = self.forward['sol']['data']
        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)

        self.noise_cov = noise_cov
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, max_iter=1000) -> mne.SourceEstimate:

        source_mat = self.fista_wrap(evoked.data, max_iter=max_iter)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    
    def fista_wrap(self, y_mat, max_iter=1000):
        srcs = []
        for y in y_mat.T:
            srcs.append ( self.fista(y, max_iter=max_iter) )
        return np.stack(srcs, axis=1)


    def fista(self, y, max_iter=1000):
        ''' The FISTA algorithm based on [1].

        Parameters 
        ---------- 
        y : numpy.ndarray
            The observations (i.e., eeg/meg data at single time point)
        max_iter : int
            Maximum number of iterations for the FISTA algorithm.
        
        References
        ----------
        [1] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
        algorithm for linear inverse problems. SIAM journal on imaging sciences,
        2(1), 183-202.
        '''
        _, n_dipoles = self.leadfield.shape
        beta = 1 / np.sum(self.leadfield**2)
        lam = 1e-14
        patience = 1000
        x_t = np.zeros(n_dipoles)
        x_t_prev = np.zeros(n_dipoles)
        x_best = np.zeros(n_dipoles)
        error_best = np.inf
        errors = []
        A_H = np.matrix(self.leadfield).getH()
        for t in range(max_iter):
            v_t = y - self.leadfield @ x_t
            
            r = x_t + beta * A_H @ v_t + ((t-2)/(t+1)) * (x_t - x_t_prev)
            x_tplus = self.soft_thresholding(r, lam)
            
            x_t_prev = deepcopy(x_t)
            x_t = x_tplus
            error = np.sum((y - self.leadfield @ x_t)**2) * 0.5 + lam * abs(x_t).sum()
            errors.append( error )

            if errors[-1] < error_best:
                x_best = deepcopy(x_t)
                error_best = errors[-1]
            if t>patience and  (np.any(np.isnan(x_tplus))  or np.all(np.array(errors[-patience:-1]) < errors[-1] )):
                break

            if self.verbose>0:
                if np.mod(t, 1000) == 0:
                    print(f"iter {t} error {errors[-1]} maxval {abs(x_t).max()}")
        if self.verbose>0:
            print(f"Finished after {t} iterations, error = {error_best}")
        return x_best
        

    @staticmethod
    def soft_thresholding(r, lam):
        r = np.squeeze(np.array(r))
        C = np.sign(r) * np.clip(abs(r) - lam, a_min=0, a_max=None)
        return C
