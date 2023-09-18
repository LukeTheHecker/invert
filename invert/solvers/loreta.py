import numpy as np
from copy import deepcopy
import mne
from scipy.sparse.csgraph import laplacian
from scipy.linalg import sqrtm, pinv
from scipy.sparse import csr_matrix
from scipy import sparse as sp
# from ..invert import BaseSolver, InverseOperator
# from .. import invert
from .base import BaseSolver, InverseOperator


class SolverLORETA(BaseSolver):
    ''' Class for the Low Resolution Tomography (LORETA) inverse solution.
    
    Attributes
    ----------

    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
    inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.

    '''
    def __init__(self, name="Low Resolution Tomography", **kwargs):
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
        leadfield = self.leadfield
        LTL = leadfield.T @ leadfield
        B = np.diag(np.linalg.norm(leadfield, axis=0))
        adjacency = mne.spatial_src_adjacency(forward['src'], verbose=self.verbose).toarray()
        laplace_operator = laplacian(adjacency)
        BLapTLapB = B @ laplace_operator.T @ laplace_operator @ B

   
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = np.linalg.inv(LTL + (alpha) * BLapTLapB) @ leadfield.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverSLORETA(BaseSolver):
    ''' Class for the standardized Low Resolution Tomography (sLORETA) inverse
        solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Pascual-Marqui, R. D. (2002). Standardized low-resolution brain
    electromagnetic tomography (sLORETA): technical details. Methods Find Exp
    Clin Pharmacol, 24(Suppl D), 5-12.
    '''
    def __init__(self, name="Standardized Low Resolution Tomography", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha=0.01, verbose=0, **kwargs):
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
        if alpha == "auto":
            msg = "sLORETA does not work well for automated regularization. Please use a floating points number (e.g., alpha=0.01)."
            raise AttributeError(msg)
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        n_chans = leadfield.shape[0]
        
        LLT = leadfield @ leadfield.T
        
        I = np.identity(n_chans)
        # one = np.ones((n_chans, 1))
        # H = I - (one @ one.T) / (one.T @ one)

        inverse_operators = []
        for alpha in self.alphas:
            # according to Grech et al 2008
            K_MNE = leadfield.T @ np.linalg.pinv(LLT + alpha *I)
            W_diag = np.sqrt(np.diag(K_MNE @ leadfield))
            W_slor = (K_MNE.T / W_diag).T

            # according to pascual-marqui 2002
            # W_slor = leadfield.T @ H @ np.linalg.pinv(H @ LLT @ H + alpha * H)
            # J = leadfield.T @ np.linalg.pinv(LLT + alpha * H)
            # S = leadfield.T @ np.linalg.pinv(LLT + alpha * H) @ leadfield
            # W_slor = J.T @ np.linalg.inv(S) @ J
            # print(J.shape, S.shape, W_slor.shape)

            # According to pascual-marqui 2009 (?)
            # C = LLT + alpha*I
            # LTC = leadfield.T @ C
            # W_slor = np.linalg.pinv(LTC @ leadfield) @ LTC
            
            inverse_operator = W_slor
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverELORETA(BaseSolver):
    ''' Class for the exact Low Resolution Tomography (eLORETA) inverse
        solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Pascual-Marqui, R. D. (2007). Discrete, 3D distributed, linear imaging
    methods of electric neuronal activity. Part 1: exact, zero error
    localization. arXiv preprint arXiv:0710.3341.

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
        stop_crit : float
            The convergence criterion to optimize the weight matrix. 
        max_iter : int
            The stopping criterion to optimize the weight matrix.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        leadfield = self.leadfield
        n_chans = leadfield.shape[0]
        # noise_cov = np.identity(n_chans)
        
        # Some pre-calculations
        I = np.identity(n_chans)
        
        inverse_operators = []
        for alpha in self.alphas:
            
            W = self.calc_W(alpha, max_iter=max_iter, stop_crit=stop_crit)
            W_inv = sp.linalg.inv(W)

            inverse_operator = W_inv @ leadfield.T @ pinv(leadfield @ W_inv @ leadfield.T + alpha * I)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def calc_W(self, alpha, max_iter=100, stop_crit=1e-3):
        from scipy.sparse import csr_matrix
        K = self.leadfield
        n_chans, n_dipoles= K.shape

        I = csr_matrix(np.identity(n_chans))
        W = csr_matrix(np.identity(n_dipoles))
        W_inv = W

        # Refine W iteratively
        for iter in range(max_iter):            
            W_old = deepcopy(W)
            W_inv = sp.linalg.inv(W)
                
            M = pinv(K @ W_inv @ K.T + alpha*I)

            W = csr_matrix(np.diag(np.sqrt(np.einsum('ij,jk,ki->i', K.T, M, K))))

            change = np.trace(abs(W.toarray()-W_old.toarray()))
            print(f"iter {iter}: {change}")
            if change < stop_crit:
                # converged!
                break

        return W


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