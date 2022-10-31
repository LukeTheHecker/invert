from tabnanny import verbose
import numpy as np
from copy import deepcopy
import mne
from scipy.sparse.csgraph import laplacian
# from ..invert import BaseSolver, InverseOperator
# from .. import invert
from .base import BaseSolver, InverseOperator


class SolverLORETA(BaseSolver):
    ''' Class for the Low Resolution Tomography (LORETA) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Low Resolution Tomography", **kwargs):
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
        leadfield = self.forward['sol']['data']
        LTL = leadfield.T @ leadfield
        B = np.diag(np.linalg.norm(leadfield, axis=0))
        adjacency = mne.spatial_src_adjacency(forward['src'], verbose=verbose).toarray()
        laplace_operator = laplacian(adjacency)
        BLapTLapB = B @ laplace_operator.T @ laplace_operator @ B

   
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = np.linalg.inv(LTL + alpha * BLapTLapB) @ leadfield.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

class SolverSLORETA(BaseSolver):
    ''' Class for the standardized Low Resolution Tomography (sLORETA) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Standardized Low Resolution Tomography", **kwargs):
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
        leadfield = self.forward['sol']['data']
        n_chans = leadfield.shape[0]
        
        LLT = leadfield @ leadfield.T
        I = np.identity(n_chans)
        one = np.ones((n_chans, 1))
        H = I - (one @ one.T) / (one.T @ one)
        

        
        inverse_operators = []
        for alpha in self.alphas:
            # according to Grech et al 2008
            # K_MNE = leadfield.T @ np.linalg.inv(LLT + alpha * np.identity(n_chans))
            # W_diag = 1 / np.diag(K_MNE @ leadfield)
            # W_slor = np.diag(W_diag)
            # W_slor = np.sqrt(W_slor)
            
            # according to pascual-marqui 2002
            T = leadfield.T @ H @ np.linalg.pinv(H @ LLT @ H + alpha * H)
            

            inverse_operator = T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)


class SolverELORETA(BaseSolver):
    ''' Class for the exact Low Resolution Tomography (eLORETA) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Exact Low Resolution Tomography", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha='auto', verbose=0, stop_crit=0.005, max_iter=100, **kwargs):
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
        leadfield = self.forward['sol']['data']
        n_chans = leadfield.shape[0]
        # noise_cov = np.identity(n_chans)
        
        # Some pre-calculations
        I = np.identity(n_chans)
        one = np.ones((n_chans, 1))
        H = I - (one @ one.T) / (one.T @ one)
        W_MNE = np.diag(np.linalg.norm(leadfield, axis=0))
        W_MNE_inv = np.linalg.inv(W_MNE)
        
        # No regularization leads to weird results with eLORETA
        if self.alphas[0] == 0:
            self.alphas[0] = 0.01
        inverse_operators = []
        for alpha in self.alphas:
            
            W = self.calc_W(H, W_MNE, W_MNE_inv, alpha, max_iter=max_iter, stop_crit=stop_crit)

            inverse_operator = np.linalg.inv(W) @ leadfield.T @ np.linalg.pinv(leadfield @ np.linalg.inv(W) @ leadfield.T + alpha * H)
            
            # According to Grech 2008:
            # D = calc_eloreta_D2(leadfield, noise_cov, alpha, stop_crit=stop_crit, verbose=verbose)
            # D_inv = np.linalg.inv(D)
            # inverse_operator = D_inv @ leadfield.T @ np.linalg.inv( leadfield @ D_inv @ leadfield.T + alpha * noise_cov )

            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)
    
    def calc_W(self, H, W_MNE, W_MNE_inv, alpha, max_iter=100, stop_crit=0.005):
        n_chans, n_dipoles = self.leadfield.shape
        
        MM = np.linalg.pinv(self.leadfield @ W_MNE_inv @ self.leadfield.T + alpha * H)
        W_last = np.zeros((n_dipoles, n_dipoles))
        # changes = [1e99,]
        # norms = [1e99,]
        # eps = 1e-16
        for i in range(max_iter):
            W_i = self.leadfield.T @ MM @ self.leadfield
            W_i = np.sqrt(np.diag(np.diagonal(W_i)))
            w_change = np.linalg.norm(W_i - W_last)
            
            if w_change < stop_crit:
                break    
            W_last = deepcopy(W_i)
            
        return W_i


def calc_eloreta_D2(leadfield, noise_cov, alpha, stop_crit=0.005, verbose=0):
    ''' Algorithm that optimizes weight matrix D as described in 
        Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
        https://www.sciencedirect.com/science/article/pii/S1053811920309150
        '''
    n_chans, n_dipoles = leadfield.shape
    # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
    D = np.identity(n_dipoles)

    if verbose>0:
        print('Optimizing eLORETA weight matrix W...')
    cnt = 0
    while True:
        old_D = deepcopy(D)
        if verbose>0:
            print(f'\trep {cnt+1}')
        D_inv = np.linalg.inv(D)
        inner_term = np.linalg.inv(leadfield @ D_inv @ leadfield.T + alpha**2*noise_cov)
            
        for v in range(n_dipoles):
            D[v, v] = np.sqrt( leadfield[:, v].T @ inner_term @ leadfield[:, v] )
        
        averagePercentChange = np.abs(1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D))))
        
        if verbose>0:
            print(f'averagePercentChange={100*averagePercentChange:.2f} %')

        if averagePercentChange < stop_crit:
            if verbose>0:
                print('\t...converged...')
            break
        cnt += 1
    if verbose>0:
        print('\t...done!')
    return D