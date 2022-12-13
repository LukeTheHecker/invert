import numpy as np
import mne
from copy import deepcopy
from ..util import find_corner
from .base import BaseSolver, InverseOperator

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
    def __init__(self, name="Minimum Variance Adaptive Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience

        '''

        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        y = evoked.data
        y -= y.mean(axis=0)
        R_inv = np.linalg.inv(y@y.T)
        leadfield -= leadfield.mean(axis=0)
        self.get_alphas(reference=y@y.T)
  
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = 1/(leadfield.T @ R_inv @ leadfield + alpha * np.identity(n_dipoles)) @ leadfield.T @ R_inv

            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

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
    def __init__(self, name="LCMV Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', weight_norm=True, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
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

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape
        
        y = evoked.data
        y -= y.mean(axis=0)

        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        self.alphas = np.logspace(-4, 1, self.n_reg_params) * np.diagonal(y@y.T).mean()

        inverse_operators = []
        for alpha in self.alphas:
            C = y@y.T + alpha*I
            C_inv = np.linalg.inv(C)

            W = (C_inv @ leadfield) / np.diagonal(leadfield.T @ C_inv @ leadfield)

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)
            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

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
    def __init__(self, name="SMV Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, weight_norm=True, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        self.weight_norm = weight_norm

        y = evoked.data
        y -= y.mean(axis=0)
        I = np.identity(n_chans)
        
        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)
            W = (C_inv @ leadfield) / np.diagonal(np.sqrt(leadfield.T @ C_inv @ leadfield))
            
            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)
            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

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
    def __init__(self, name="WNMV Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, weight_norm=True, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape

        self.weight_norm = weight_norm
        y = evoked.data
        y -= y.mean(axis=0)
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

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
    def __init__(self, name="HOCMV Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, weight_norm=True, alpha='auto', order=3, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
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

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = self.leadfield.shape
        
        self.weight_norm = weight_norm
        
        y = evoked.data
        y -= y.mean(axis=0)
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

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
    def __init__(self, name="ESMV Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape
        

        y = evoked.data
        y -= y.mean(axis=0)
        leadfield -= leadfield.mean(axis=0)
        I = np.identity(n_chans)
        
        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)

        U, s, _ = np.linalg.svd(C)
        # j = find_corner(np.arange(len(s)), s)
        
        # Find number of Signal Subspaces:
        j = np.where(((s**2)*len((s**2)) / (s**2).sum()) < np.exp(-16))[0][0]

        Us = U[:, :j]
        # Un = U[:, j:]

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)

            W_mv = (C_inv @ leadfield) / np.diagonal(leadfield.T @ C_inv @ leadfield)
            W = Us @ Us.T @ W_mv

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

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
    def __init__(self, name="MCMV Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, weight_norm=True, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape

        self.weight_norm = weight_norm

        y = evoked.data
        y -= y.mean(axis=0)
        leadfield -= leadfield.mean(axis=0)
        I = np.identity(n_chans)

        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=C)

        inverse_operators = []
        for alpha in self.alphas:
            C_inv = np.linalg.inv(C + alpha * I)

            W = C_inv @ leadfield * np.diagonal(np.linalg.inv(leadfield.T @ C_inv @ leadfield))

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)



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
    def __init__(self, name="ReciPSIICOS", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, weight_norm=True, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        leadfield /= np.linalg.norm(leadfield, axis=0)
        n_chans, n_dipoles = leadfield.shape
        
        self.weight_norm = weight_norm
        
        y = evoked.data
        y -= y.mean(axis=0)

        
        # Step 1
        G_pwr = leadfield @ leadfield.T
        
        # Step 2
        U_pwr, S_pwr, _ = np.linalg.svd(G_pwr, full_matrices=False)
        k = find_corner(np.arange(len(S_pwr)), S_pwr)
        P = U_pwr[:, :k] @ U_pwr[:, :k].T
        
        
        I = np.identity(n_chans)
        # Recompute regularization based on the max eigenvalue of the Covariance
        # Matrix (opposed to that of the leadfield)
        C = y@y.T
        self.alphas = self.get_alphas(reference=G_pwr)
        # self.alphas = np.logspace(-4, 1, self.n_reg_params) * np.diagonal(y@y.T).mean()
        
        inverse_operators = []
        for alpha in self.alphas:
            C = y@y.T + alpha * I

            # Step 3
            C_x = (P@C).T

            # Step 4
            E, A, _ = np.linalg.svd(C_x, full_matrices=False)

            # Make new Covariance positive semidefinite
            C_x = E @ np.diag(np.abs(A)) @ E.T
            C_x_inv = np.linalg.inv(C_x)

            W = (C_x_inv @ leadfield) * np.diagonal(np.linalg.inv(leadfield.T @ C_x_inv @ leadfield))

            if self.weight_norm:
                W /= np.linalg.norm(W, axis=0)

            inverse_operator = W.T
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)


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
    def __init__(self, name="SAM Beamformer", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, weight_norm=True, alpha='auto', verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.Evoked
            The evoked data object.
        weight_norm : bool
            Normalize the filter weight matrix W to unit length of the columns.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        self.weight_norm = weight_norm
        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        n_chans, n_dipoles = leadfield.shape
        

        y = evoked.data
        y -= y.mean(axis=0)
        leadfield -= leadfield.mean(axis=0)
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

