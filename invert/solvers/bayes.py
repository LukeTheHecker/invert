import numpy as np
import mne
from copy import deepcopy
from scipy.sparse.csgraph import laplacian
from scipy.sparse import coo_matrix
from .base import BaseSolver, InverseOperator

class SolverBCS(BaseSolver):
    ''' Class for the Bayesian Compressed Sensing (BCS) inverse solution [1].
    
    Attributes
    ----------
    
    
    References
    ----------
    [1] Ji, S., Xue, Y., & Carin, L. (2008). Bayesian compressive sensing. IEEE
    Transactions on signal processing, 56(6), 2346-2356.

    '''
    def __init__(self, name="Bayesian Compressed Sensing", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", **kwargs):
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
        self.leadfield_norm = self.leadfield

        return self

    def apply_inverse_operator(self, mne_obj, max_iter=100, alpha_0=0.01, eps=1e-16) -> mne.SourceEstimate:
        ''' Apply the inverse operator.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        max_iter : int
            Maximum number of iterations
        alpha_0 : float
            Regularization parameter
        eps : float
            Epsilon, used to avoid division by zero.
        
        Return
        ------
        stc : mne.SourceEstimate
            The SourceEstimate data structure containing the inverse solution.

        '''
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.calc_bcs_solution(data, max_iter=max_iter, alpha_0=alpha_0, eps=eps)
        stc = self.source_to_object(source_mat)
        return stc
    
    def calc_bcs_solution(self, y, max_iter=100, alpha_0=0.01, eps=1e-16):
        ''' This function computes the BCS inverse solution.

        Parameters
        ----------
        y : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)
        max_iter : int
            Maximum number of iterations
        alpha_0 : float
            Regularization parameter
        eps : float
            Epsilon, used to avoid division by zero.
        
        Return
        ------
        x_hat : numpy.ndarray
            The source estimate.
        '''

        alpha_0 = np.clip(alpha_0, a_min=1e-6, a_max=None)
        n_chans, _ = y.shape
        n_dipoles = self.leadfield_norm.shape[1]
        
        # preprocessing
        y -= y.mean(axis=0)
        
        alphas = np.ones(n_dipoles)
        D = np.diag(alphas)

        LLT = self.leadfield_norm.T @ self.leadfield_norm
        sigma = np.linalg.inv(alpha_0 * LLT + D)
        mu = alpha_0 * sigma @ self.leadfield_norm.T @ y
        proj_norm = self.leadfield_norm.T @ y
        proj = self.leadfield.T @ y
        
        residual_norms = [1e99]
        x_hats = []
        for i in range(max_iter):
            gammas = np.array([1 - alphas[ii] * sigma[ii,ii] for ii in range(n_dipoles)])
            gammas[np.isnan(gammas)] = 0

            
            alphas = gammas / np.linalg.norm(mu**2, axis=1)
            alpha_0 = 1 / ( np.linalg.norm(y - self.leadfield_norm @ mu) / (n_chans - gammas.sum()) )
            D = np.diag(alphas) + eps
            sigma = np.linalg.inv(alpha_0 * LLT + D)
            mu = alpha_0 * sigma @ proj_norm

            # var = 1/alpha_0
            # try:
            #     D_inv = np.linalg.inv(D)
            # except:
            #     break
            # C = var**2 * I + self.leadfield @ D_inv @ self.leadfield.T
            
            Gamma = np.diag(gammas)
            x_hat = Gamma @ proj
            residual_norm = np.linalg.norm(y - self.leadfield @ x_hat)
            # print(residual_norm)
            if residual_norm > residual_norms[-1]:
                x_hat = x_hats[-1]
                # print(f"Stopping after {i} iterations.")
                break
            residual_norms.append(residual_norm)
            x_hats.append(x_hat)
            # print("residual_norm: ", residual_norm)

        return x_hat

class SolverGammaMAP(BaseSolver):
    ''' Class for the Gamma Maximum A Posteriori (Gamma-MAP) inverse solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG
    source imaging. NeuroImage, 44(3), 947-966.

    '''
    def __init__(self, name="Gamma-MAP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", smoothness_prior=False,
                              max_iter=100, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : str/ float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        smoothness_prior : bool

        
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape
        data = self.unpack_data_obj(mne_obj)

        if smoothness_prior:
            adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
            self.gradient = laplacian(adjacency).toarray().astype(np.float32)
            self.sigma_s = np.identity(n_dipoles) @ abs(self.gradient)
        else:
            self.gradient = None
            self.sigma_s = np.identity(n_dipoles)
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_gamma_map_inverse_operator(data, alpha, max_iter=max_iter)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    
    def make_gamma_map_inverse_operator(self, B, alpha, max_iter=100):
        ''' Computes the gamma MAP inverse operator based on the M/EEG data.
        
        Parameters
        ----------
        B : numpy.ndarray
            The M/EEG data matrix (channels, time points).
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        
        Return
        ------
        inverse_operator : numpy.ndarray
            The inverse operator which can be used to compute inverse solutions from new data.

        '''
        L = deepcopy(self.leadfield)
        db, n = B.shape
        ds = L.shape[1]

        # Ensure Common average reference
        B -= B.mean(axis=0)
        L -= L.mean(axis=0)
        
        # Data Covariance Matrix
        gammas = np.ones(ds)
        sigma_e = alpha * np.identity(db)  
        
        sigma_b = sigma_e + L @ self.sigma_s @ L.T
        sigma_b_inv = np.linalg.inv(sigma_b)
        
        for k in range(max_iter):
            old_gammas = deepcopy(gammas)
            
            # according to equation (30)
            term_1 = (gammas/np.sqrt(n)) * np.sqrt(np.sum((L.T @ sigma_b_inv @ B )**2, axis=1))
            term_2 = 1 / np.sqrt(np.diagonal((L.T @ sigma_b_inv @ L )))

            gammas = term_1 * term_2
            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break

        gammas_final = gammas / gammas.max()
        sigma_s_hat = np.diag(gammas_final) @ self.sigma_s  #  np.array([gammas_final[i] * C[i] for i in range(ds)])
        inverse_operator = sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)
        
        # This way the inverse operator would be applied to M/EEG matrix B:
        # S = inverse_operator @ B

        return inverse_operator
        
    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x@x.T))

class SolverSourceMAP(BaseSolver):
    ''' Class for the Source Maximum A Posteriori (Source-MAP) inverse solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG
    source imaging. NeuroImage, 44(3), 947-966.

    '''
    def __init__(self, name="Source-MAP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", smoothness_prior=False,
                              max_iter=100, p=0.5, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        p : 0 < p < 2 
            Hyperparameter which controls sparsity. Default: p = 0.5
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_chans, n_dipoles = leadfield.shape

        if smoothness_prior:
            adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
            self.gradient = laplacian(adjacency).toarray().astype(np.float32)
            self.sigma_s = np.identity(n_dipoles) @ abs(self.gradient)
        else:
            self.gradient = None
            self.sigma_s = np.identity(n_dipoles)

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_source_map_inverse_operator(data, alpha, max_iter=max_iter, p=p)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    
    def make_source_map_inverse_operator(self, B, alpha, max_iter=100, p=0.5):
        ''' Computes the source MAP inverse operator based on the M/EEG data.
        
        Parameters
        ----------
        B : numpy.ndarray
            The M/EEG data matrix (channels, time points).
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        p : 0 < p < 2 
            Hyperparameter which controls sparsity. Default: p = 0.5

        
        Return
        ------
        inverse_operator : numpy.ndarray
            The inverse operator which can be used to compute inverse solutions from new data.

        '''

        L = deepcopy(self.leadfield)
        db, n = B.shape
        ds = L.shape[1]

        # Ensure Common average reference
        B -= B.mean(axis=0)
        L -= L.mean(axis=0)
        L /= np.linalg.norm(L, axis=0)
 
        # Data Covariance Matrix
        # Cb = B @ B.T
        gammas = np.ones(ds)
        sigma_e = alpha * np.identity(db)

        sigma_b = sigma_e + L @ self.sigma_s @ L.T
        sigma_b_inv = np.linalg.inv(sigma_b)
        
        for k in range(max_iter):
            # print(k)
            old_gammas = deepcopy(gammas)
            
            gammas = ((1/n) * np.sqrt(np.sum(( np.diag(gammas) @ L.T @ sigma_b_inv @ B )**2, axis=1)))**((2-p)/2)

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break
            # gammas /= np.linalg.norm(gammas)

        gammas_final = gammas / gammas.max()
        sigma_s_hat = np.diag(gammas_final) @ self.sigma_s  #  np.array([gammas_final[i] * C[i] for i in range(ds)])
        inverse_operator = sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)

        # This way the inverse operator would be applied to M/EEG matrix B:
        # S = inverse_operator @ B

        return inverse_operator
        
    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x@x.T))

class SolverGammaMAPMSP(BaseSolver):
    ''' Class for the Gamma Maximum A Posteriori (Gamma-MAP) inverse solution
    using multiple sparse priors (MSP).
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for MEG/EEG
    source imaging. NeuroImage, 44(3), 947-966.
    
    '''
    def __init__(self, name="Gamma-MAP-MSP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", 
                              max_iter=100, p=0.5, smoothness_order=1, verbose=0, 
                              **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA-like solution.
        p : 0 < p < 2 
            Hyperparameter which controls sparsity. Default: p = 0
        smoothness_order : int
            Controls the smoothness prior. The higher this integer, the higher
            the pursued smoothness of the inverse solution.

        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_chans, _ = leadfield.shape
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_source_map_inverse_operator(data, alpha, 
                                                                    max_iter=max_iter, p=p, 
                                                                    smoothness_order=smoothness_order)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def make_source_map_inverse_operator(self, B, alpha, max_iter=100, p=0.5, smoothness_order=1):
        ''' Computes the source MAP inverse operator based on the M/EEG data.
        
        Parameters
        ----------
        B : numpy.ndarray
            The M/EEG data matrix (channels, time points).
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        p : 0 < p < 2 
            Hyperparameter which controls sparsity. Default: p = 0.5
        smoothness_order : int
            Controls the smoothness prior. The higher this integer, the higher
            the pursued smoothness of the inverse solution.
        
        Return
        ------
        inverse_operator : numpy.ndarray
            The inverse operator which can be used to compute inverse solutions from new data.

        '''
        L = deepcopy(self.leadfield)
        db, n = B.shape
        ds = L.shape[1]

        # Ensure Common average reference
        B -= B.mean(axis=0)
        L -= L.mean(axis=0)
        
 
        # Data Covariance Matrix
        # Cb = B @ B.T
        L_smooth, gradient = self.get_smooth_prior_cov(L, smoothness_order)
        gammas = np.ones(ds)
        sigma_e = alpha * np.identity(db)  
        sigma_s = np.identity(ds) # identity leads to weighted minimum L2 Norm-type solution
        sigma_b = sigma_e + L_smooth @ sigma_s @ L_smooth.T
        sigma_b_inv = np.linalg.inv(sigma_b)
        
        for k in range(max_iter):
            # print(k)
            old_gammas = deepcopy(gammas)
            
            # gammas = ((1/n) * np.sqrt(np.sum(( np.diag(gammas) @ L_smooth.T @ sigma_b_inv @ B )**2, axis=1)))**((2-p)/2)
            
            term_1 = (gammas/np.sqrt(n)) * np.sqrt(np.sum((L_smooth.T @ sigma_b_inv @ B )**2, axis=1))
            term_2 = 1 / np.sqrt(np.diagonal((L_smooth.T @ sigma_b_inv @ L_smooth )))
            gammas = term_1 * term_2

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break
            # print(gammas.min(), gammas.max())
            # gammas /= np.linalg.norm(gammas)
        
        # Smooth gammas according to smooth priors
        gammas_final = abs(gammas@gradient)
        gammas_final = gammas / gammas.max()

        sigma_s_hat = np.diag(gammas_final) @ sigma_s
        inverse_operator = sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)
        # S = inverse_operator @ B
        return inverse_operator
        
    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x@x.T))
    
    def get_smooth_prior_cov(self, L, smoothness_order):
        ''' Create a smooth prior on the covariance matrix.
        
        Parameters
        ----------
        L : numpy.ndarray
            Leadfield matrix (channels, dipoles)
        smoothness_order : int
            The higher the order, the smoother the prior.

        Return
        ------
        L : numpy.ndarray
            The smoothed Leadfield matrix (channels, dipoles)
        gradient : numpy.ndarray
            The smoothness gradient (laplacian matrix)

        '''
        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        gradient = laplacian(adjacency).toarray().astype(np.float32)
        
        for _ in range(smoothness_order):
            gradient = gradient @ gradient
        L = L @ abs(gradient)
        L -= L.mean(axis=0)
        # L /= np.linalg.norm(L, axis=0)
        return L, gradient

class SolverSourceMAPMSP(BaseSolver):
    ''' Class for the Source Maximum A Posteriori (Source-MAP) inverse solution
    using multiple sparse priors [1]. The method is conceptually similar to [2],
    but formally not equal.
    
    Attributes
    ----------

    
    References
    ----------
    [1] Wipf, D., & Nagarajan, S. (2009). A unified Bayesian framework for
    MEG/EEG source imaging. NeuroImage, 44(3), 947-966. 
    
    [2] Friston, K., Harrison, L., Daunizeau, J., Kiebel, S., Phillips, C.,
    Trujillo-Barreto, N., ... & Mattout, J. (2008). Multiple sparse priors for
    the M/EEG inverse problem. NeuroImage, 39(3), 1104-1120.
    
    '''
    def __init__(self, name="Source-MAP-MSP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", 
                              max_iter=100, p=0.5, smoothness_order=1, verbose=0, 
                              **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        p : 0 < p < 2 
            Hyperparameter which controls sparsity. Default: p = 0.5
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        smoothness_order : int
            Controls the smoothness prior. The higher this integer, the higher
            the pursued smoothness of the inverse solution.

        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        leadfield = self.leadfield
        n_chans, _ = leadfield.shape
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_source_map_inverse_operator(data, alpha, 
                                                                    max_iter=max_iter, p=p, 
                                                                    smoothness_order=smoothness_order)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    
    def make_source_map_inverse_operator(self, B, alpha, max_iter=100, p=0.5, smoothness_order=1):
        ''' Computes the source MAP inverse operator based on the M/EEG data.
        
        Parameters
        ----------
        B : numpy.ndarray
            The M/EEG data matrix (channels, time points).
        alpha : float
            The regularization parameter.
        max_iter : int
            Maximum numbers of iterations to find the optimal hyperparameters.
            max_iter = 1 corresponds to sLORETA.
        p : 0 < p < 2 
            Hyperparameter which controls sparsity. Default: p = 0.5
        smoothness_order : int
            Controls the smoothness prior. The higher this integer, the higher
            the pursued smoothness of the inverse solution.
        
        Return
        ------
        inverse_operator : numpy.ndarray
            The inverse operator which can be used to compute inverse solutions from new data.

        '''

        L = deepcopy(self.leadfield)
        db, n = B.shape
        ds = L.shape[1]

        # Ensure Common average reference
        B -= B.mean(axis=0)
        L -= L.mean(axis=0)
        
 
        # Data Covariance Matrix
        # Cb = B @ B.T
        L_smooth, gradient = self.get_smooth_prior_cov(L, smoothness_order)
        gammas = np.ones(ds)
        sigma_e = alpha * np.identity(db)  
        sigma_s = np.identity(ds) # identity leads to weighted minimum L2 Norm-type solution
        sigma_b = sigma_e + L_smooth @ sigma_s @ L_smooth.T
        sigma_b_inv = np.linalg.inv(sigma_b)
        
        for k in range(max_iter):
            # print(k)
            old_gammas = deepcopy(gammas)
            
            gammas = ((1/n) * np.sqrt(np.sum(( np.diag(gammas) @ L_smooth.T @ sigma_b_inv @ B )**2, axis=1)))**((2-p)/2)

            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break
            # print(gammas.min(), gammas.max())
            # gammas /= np.linalg.norm(gammas)
        
        # Smooth gammas according to smooth priors
        gammas_final = abs(gammas@gradient)
        gammas_final = gammas / gammas.max()

        sigma_s_hat = np.diag(gammas_final) @ sigma_s  #  np.array([gammas_final[i] * C[i] for i in range(ds)])
        inverse_operator = sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)
        # S = inverse_operator @ B
        return inverse_operator
        
    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x@x.T))
    
    def get_smooth_prior_cov(self, L, smoothness_order):
        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        gradient = laplacian(adjacency).toarray().astype(np.float32)
        
        for i in range(smoothness_order):
            gradient = gradient @ gradient
        L = L @ abs(gradient)
        L -= L.mean(axis=0)
        # L /= np.linalg.norm(L, axis=0)
        return L, gradient