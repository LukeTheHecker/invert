import numpy as np
import mne
from copy import deepcopy
from ..util import find_corner
from .base import BaseSolver, InverseOperator
from time import time

class SolverMVAB(BaseSolver):
    ''' Class for the Minimum Variance Adaptive Beamformer (MVAB) inverse
        solution [1].
    
    Attributes
    ----------

    References
    ----------
    [1] Vorobyov, S. A. (2013). Principles of minimum variance robust adaptive
    beamforming design. Signal Processing, 93(12), 3264-3277.

    '''
    def __init__(self, name="Minimum Variance Adaptive Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience

        '''

        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        y = data
        C = y@y.T
        I = np.identity(n_chans)

        self.get_alphas(reference=C)
        
        inverse_operators = []
        for alpha in self.alphas:
            R_inv = np.linalg.inv(C + alpha * I)
            inverse_operator = 1/(leadfield.T @ R_inv @ leadfield + alpha * np.identity(n_dipoles)) @ leadfield.T @ R_inv

            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverLCMV(BaseSolver):
    ''' Class for the Linearly Constrained Minimum Variance Beamformer (LCMV)
        inverse solution [1].
    
    Attributes
    ----------

    References
    ----------
    [1] Van Veen, B. D., & Buckley, K. M. (1988). Beamforming: A versatile
    approach to spatial filtering. IEEE assp magazine, 5(2), 4-24.
    
    '''
    def __init__(self, name="LCMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', weight_norm=True, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        self.weight_norm = weight_norm
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape
        
        y = data
        
        I = np.identity(n_chans)
        C = y@y.T
        
        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        self.alphas = np.logspace(-4, 1, self.n_reg_params) * np.diagonal(y@y.T).mean()

        inverse_operators = []
        for alpha in self.alphas:
            
            C_inv = np.linalg.inv(C + alpha*I)

            # W = (C_inv @ leadfield) / np.diagonal(leadfield.T @ C_inv @ leadfield)
            upper = C_inv @ leadfield
            lower = np.einsum('ij,jk,ki->i', leadfield.T, C_inv, leadfield)
            W = upper / lower
            
            # C_inv_L = C_inv @ leadfield
            # diagonal_elements = np.einsum('ij,ji->i', leadfield.T, C_inv_L)
            # W = C_inv_L / diagonal_elements
            
            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)
            
            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self


class SolverSMV(BaseSolver):
    ''' Class for the Standardized Minimum Variance (SMV) Beamformer inverse
        solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones, R.
    (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    '''
    def __init__(self, name="SMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        self.weight_norm = weight_norm

        y = data
        I = np.identity(n_chans)
        
        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            W = (C_inv @ leadfield) / np.sqrt(np.diagonal(leadfield.T @ C_inv @ leadfield))
            
            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)
            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverWNMV(BaseSolver):
    ''' Class for the Weight-normalized Minimum Variance (WNMV) Beamformer
        inverse solution [1].
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    '''
    def __init__(self, name="WNMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        self.weight_norm = weight_norm
        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)
        
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            C_inv_2 = np.linalg.inv(C_inv)
            W = (C_inv @ leadfield) / np.sqrt(abs(np.diagonal(leadfield.T @ C_inv_2 @ leadfield)))

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverHOCMV(BaseSolver):
    ''' Class for the Higher-Order Covariance Minimum Variance (HOCMV)
        Beamformer inverse solution [1].
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    '''
    def __init__(self, name="HOCMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, alpha='auto', order=3, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        order : int
            The order of the covariance matrix. Should be a positive integer not
            evenly divisible by two {3, 5, 7, ...}
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape
        
        self.weight_norm = weight_norm
        
        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)
        
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            C_inv_n = deepcopy(C_inv)
            for _ in range(order-1):
                C_inv_n = np.linalg.inv(C_inv_n)

            W = (C_inv @ leadfield) / np.sqrt(abs(np.diagonal(leadfield.T @ C_inv_n @ leadfield)))
            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverESMV(BaseSolver):
    ''' Class for the Eigenspace-based Minimum Variance (ESMV) Beamformer
        inverse solution [1].
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.
    
    '''
    def __init__(self, name="ESMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape
        

        y = data
        I = np.identity(n_chans)
        
        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)
        subspace = self.reduce_rank_matrix(C)

        
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            C_inv_leadfield = C_inv @ leadfield
            diag_elements = np.einsum('ij,ji->i', leadfield.T, C_inv_leadfield)
            W_mv = C_inv_leadfield / diag_elements
            W = subspace @ W_mv

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def reduce_rank_matrix(self, C):
        # Calculte eigenvectors and eigenvalues
        U, s, _ = np.linalg.svd(C, full_matrices=False)
        
        # Find optimal rank using L-curve Corner approach:
        n_comp_l = find_corner(np.arange(len(s)), s)
        
        # Find optimal rank using drop-off approach:
        s_ = s / s.max()
        n_comp_d = np.where( abs(np.diff(s_)) < 0.001 )[0]
        if len(n_comp_d) > 0:
            n_comp_d = n_comp_d[0] + 2
        else:
            n_comp_d = n_comp_l
        
        # Kaiser Rule
        n_comp_k = np.where(s < np.mean(s))[0][0]
        print(n_comp_k, n_comp_l, n_comp_d)
        
        # Combine the approaches
        n_comp = np.ceil((n_comp_d + n_comp_l + n_comp_k)/3).astype(int)
        
        # Transform data
        subspace = U[:, :n_comp] @ U[:, :n_comp].T
        return subspace

class SolverMCMV(BaseSolver):
    ''' Class for the Multiple Constrained Minimum Variance (MCMV) Beamformer
    inverse solution [1].
    
    Attributes
    ----------

    
    References
    ----------
    [1] Nunes, A. S., Moiseev, A., Kozhemiako, N., Cheung, T., Ribary, U., &
    Doesburg, S. M. (2020). Multiple constrained minimum variance beamformer
    (MCMV) performance in connectivity analyses. NeuroImage, 208, 116386.

    '''
    def __init__(self, name="MCMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, noise_cov=None, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        # leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        if noise_cov is None:
            noise_cov = np.identity(n_chans)
            
        self.weight_norm = weight_norm

        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y @ y.T
        self.alphas = self.get_alphas(reference=C)
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            leadfield_C_inv = leadfield.T @ C_inv

            # Use np.einsum to compute the diagonal elements
            diag_elements = np.einsum('ij,ji->i', leadfield_C_inv, leadfield)

            W = C_inv @ leadfield * (1.0 / diag_elements)
            # W = C_inv @ leadfield @ np.linalg.pinv(leadfield.T @ C_inv @ leadfield)

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]

        return self
    
class SolverUnitNoiseGain(BaseSolver):
    ''' Class for the Unit Noise Gain (UNIG) Beamformer
    inverse solution [1].
    
    Attributes
    ----------

    
    References
    ----------
    [1] 
    '''
    def __init__(self, name="UNIG Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, noise_cov=None, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        # leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        if noise_cov is None:
            noise_cov = np.identity(n_chans)
            
        self.weight_norm = weight_norm

        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y @ y.T
        self.alphas = self.get_alphas(reference=C)
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            C_inv_inv = np.linalg.inv(C_inv)
            leadfield_C_inv_inv = leadfield.T @ C_inv_inv

            # Use np.einsum to compute the diagonal elements
            diag_elements = np.einsum('ij,ji->i', leadfield_C_inv_inv, leadfield)

            # W = C_inv @ leadfield * (1.0 / diag_elements)
            W = C_inv @ leadfield * (1 / np.sqrt(diag_elements))

            # W = C_inv @ leadfield @ np.linalg.pinv(leadfield.T @ C_inv @ leadfield)

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]

        return self

class SolverHOCMCMV(BaseSolver):
    ''' Class for the Higher-Order Covariance Multiple Constrained Minimum Variance (HOCMCMV)
        Beamformer inverse solution [1].
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Jonmohamadi, Y., Poudel, G., Innes, C., Weiss, D., Krueger, R., & Jones,
    R. (2014). Comparison of beamformers for EEG source signal reconstruction.
    Biomedical Signal Processing and Control, 14, 175-188.

    '''
    def __init__(self, name="HOCMCMV Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, alpha='auto', order=3, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        order : int
            The order of the covariance matrix. Should be a positive integer not
            evenly divisible by two {3, 5, 7, ...}
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape
        
        self.weight_norm = weight_norm
        
        y = data
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)
        
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            C_inv_n = deepcopy(C_inv)
            for _ in range(order-1):
                C_inv_n = np.linalg.inv(C_inv_n)
            W = C_inv @ leadfield * np.diagonal(np.linalg.inv(leadfield.T @ C_inv_n @ leadfield))
            
            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    

class SolverReciPSIICOS(BaseSolver):
    ''' Class for the Reciprocal Phase Shift Invariant Imaging of Coherent
    Sources (ReciPSIICOS) Beamformer inverse solution [1].
    
    Attributes
    ----------


    References
    ----------    
    [1] Kuznetsova, A., Nurislamova, Y., & Ossadtchi, A. (2021). Modified
    covariance beamformer for solving MEG inverse problem in the environment
    with correlated sources. Neuroimage, 228, 117677.

    '''
    def __init__(self, name="ReciPSIICOS", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, K=3, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        leadfield = self.leadfield
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape
        
        self.weight_norm = weight_norm
        
        y = data
        C = y@y.T

        # Step 1
        G_pwpr = self.construct_G_pwpr(leadfield.T)
        
        # Step 2
        P = self.compute_projector(G_pwpr, K)

        # Step 3
        C_projected = self.apply_projection(P, C)

        # Step 4: SVD of C_projected and reconstruct with absolute eigenvalues
        E, A, _ = np.linalg.svd(C_projected, full_matrices=False)
        C = E @ np.diag(np.abs(A)) @ E.T
        
        I = np.identity(n_chans)
        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            leadfield_C_inv = leadfield.T @ C_inv

            # MCMV
            diag_elements = np.einsum('ij,ji->i', leadfield_C_inv, leadfield)
            W = C_inv @ leadfield * (1.0 / diag_elements)

            #LCMV
            # upper = C_inv @ leadfield
            # lower = np.einsum('ij,jk,ki->i', leadfield.T, C_inv, leadfield)
            # W = upper / lower

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    
    @staticmethod
    def construct_G_pwpr(g_list):
        # g_list is a list of topography vectors g_i for N sources
        N = len(g_list)
        q_list = [np.outer(g, g).flatten() for g in g_list]
        G_pwpr = np.column_stack(q_list)
        return G_pwpr

    @staticmethod
    def compute_projector(G_pwpr, K):
        U, S, Vt = np.linalg.svd(G_pwpr, full_matrices=False)
        U_K = U[:, :K]  # Keep only the first K singular vectors
        P = U_K @ U_K.T  # Projector matrix
        return P

    @staticmethod
    def apply_projection(P, C_x):
        # Vectorize the sensor-space covariance matrix C_x
        vec_Cx = C_x.flatten()
        # Project the vectorized covariance matrix
        projected_vec_Cx = P @ vec_Cx
        # Reshape back to matrix form
        C_x_projected = np.reshape(projected_vec_Cx, C_x.shape)
        return C_x_projected


class SolverSAM(BaseSolver):
    ''' Class for the Synthetic Aperture Magnetometry Beamformer (SAM) inverse
    solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Robinson, S. E. V. J. (1999). Functional neuroimaging by synthetic
    aperture magnetometry (SAM). Recent advances in biomagnetism.

    '''
    def __init__(self, name="SAM Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        self.weight_norm = weight_norm
        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape
        

        y = data
        I = np.identity(n_chans)
  
        
        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(y@y.T + alpha * I)
            W = []
            for i in range(n_dipoles):
                l = leadfield[:, i][:, np.newaxis]
                w = (C_inv@l) / (l.T@C_inv@l)
                W.append(w)
            W = np.stack(W, axis=1)[:, :, 0]
            if self.weight_norm:
                W = W / np.linalg.norm(W, axis=0)
            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverEBB(BaseSolver):
    """
    Empirical Bayesian Beamformer (EBB) solver for M/EEG inverse problem.
    """
    def __init__(self, name="Empirical Bayesian Beamformer", reduce_rank=True, rank="auto", **kwargs):
        self.name = name
        return super().__init__(reduce_rank=reduce_rank, rank=rank, **kwargs)
        
    def make_inverse_operator(self, forward, mne_obj, *args, weight_norm=True, noise_cov=None, alpha='auto', **kwargs):
        """
        Solve the inverse problem using the Empirical Bayesian Beamformer method.

        Parameters:
        -----------
        data : array, shape (n_channels, n_times)
            The sensor data.
        forward : array, shape (n_channels, n_sources)
            The forward solution.
        noise_cov : array, shape (n_channels, n_channels), optional
            The noise covariance matrix.

        Returns:
        --------
        sources : array, shape (n_sources, n_times)
            The estimated source time series.
        """
        print(type(forward))
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_channels, n_times = data.shape
        n_sources = leadfield.shape[1]
        leadfield /= np.linalg.norm(leadfield, axis=0)
        

        # Compute data covariance
        data_cov = np.cov(data)

        # handle noise_cov
        if noise_cov is None:
            noise_cov = np.identity(n_channels)

        # Initialize source covariance
        inverse_operators = []
        self.alphas = self.get_alphas(reference=leadfield@leadfield.T)
        for alpha in self.alphas:
            source_cov = np.eye(n_sources)
            # Iterative process
            for n_iter in range(100):  # You can adjust the number of iterations
                # Compute regularized inverse
                C = leadfield @ source_cov @ leadfield.T
                C_inv = np.linalg.inv(C + alpha * noise_cov)
                
                # Update source covariance
                W = source_cov @ leadfield.T @ C_inv
                new_source_cov = W @ data_cov @ W.T
                if weight_norm:
                    W /= np.linalg.norm(W, axis=0)
                
                # Check convergence
                if np.allclose(new_source_cov, source_cov, rtol=1e-9):
                    print(f"Converged after {n_iter} iterations")
                    break
                
                source_cov = new_source_cov

            # Compute final beamformer weights
            W = source_cov @ leadfield.T @ np.linalg.inv(leadfield @ source_cov @ leadfield.T + noise_cov)
            inverse_operators.append(W)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]

        return self