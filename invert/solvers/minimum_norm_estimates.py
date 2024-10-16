import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator
from scipy.sparse.csgraph import laplacian

class SolverMNE(BaseSolver):
    ''' Class for the Minimum Norm Estimate (MNE) inverse solution [1]. The
        formulas provided by [2] were used for implementation.

    Attributes
    ----------
    
    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
    inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.
    
    [2] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.

    '''
    def __init__(self, name="Minimum Norm Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", verbose=0, **kwargs):
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
        super().make_inverse_operator(forward, *args, reference=None, alpha=alpha, **kwargs)
        
        leadfield = self.leadfield
        n_chans, _ = leadfield.shape
        
        LLT = leadfield @ leadfield.T
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = leadfield.T @ np.linalg.inv(LLT + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverGFTMNE(BaseSolver):
    ''' Class for the Minimum Norm Estimate (MNE) inverse solution [1]. The
        formulas provided by [2] were used for implementation.

    Attributes
    ----------
    
    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
    inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.
    
    [2] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.

    '''
    def __init__(self, name="Minimum Norm Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", verbose=0, **kwargs):
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
        super().make_inverse_operator(forward, *args, reference=None, alpha=alpha, **kwargs)
        
        leadfield = self.leadfield
        n_chans, _ = leadfield.shape

        # Get Adjacency matrix
        
        adjacency = mne.spatial_src_adjacency(forward['src'], verbose=0)
        lap = laplacian(adjacency)
        
        U, eigenvalues , _ = np.linalg.svd(lap.toarray(), full_matrices=False)
        # Filter
        num_eigenvalues = len(eigenvalues)
        cutoff_index = 100  # int(num_eigenvalues * 0.3)  # 30% cutoff
        # U = U[:, :cutoff_index]
        U = np.real(U[:, -cutoff_index:])

        # Transform leadfield
        leadfield_gft = leadfield @ U

        LLT = leadfield_gft @ leadfield_gft.T
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = leadfield_gft.T @ np.linalg.inv(LLT + alpha * np.identity(n_chans))
            inverse_operators.append(U @ inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverWMNE(BaseSolver):
    ''' Class for the Weighted Minimum Norm Estimate (wMNE) inverse solution
        [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.
    '''
    def __init__(self, name="Weighted Minimum Norm Estimate", **kwargs):
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
        W = np.diag(np.linalg.norm(self.leadfield, axis=0))
        WTW = np.linalg.inv(W.T @ W)
        LWTWL = self.leadfield @ WTW @ self.leadfield.T
        n_chans, _ = self.leadfield.shape

      
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = WTW @ self.leadfield.T  @ np.linalg.inv(LWTWL + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

class SolverDSPM(BaseSolver):
    ''' Class for the Dynamic Statistical Parametric Mapping (dSPM) inverse
        solution [1,2].  The formulas provided by [3] were used for
        implementation.
    
    Attributes
    ----------
    
    References
    ----------
    [1] Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J.
    W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric
    mapping: combining fMRI and MEG for high-resolution imaging of cortical
    activity. neuron, 26(1), 55-67.

    [2] Dale, A. M., Fischl, B., & Sereno, M. I. (1999). Cortical surface-based
    analysis: I. Segmentation and surface reconstruction. Neuroimage, 9(2),
    179-194.

    [3] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.
    '''
    def __init__(self, name="Dynamic Statistical Parametric Mapping", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha=0.01, noise_cov=None, source_cov=None,
                            verbose=0, **kwargs):
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
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_chans, n_dipoles = self.leadfield.shape

        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        if source_cov is None:
            source_cov = np.identity(n_dipoles)
        

        inverse_operators = []
        leadfield_source_cov = source_cov @ self.leadfield.T
        LLS = self.leadfield @ leadfield_source_cov
        for alpha in self.alphas:
            K = leadfield_source_cov @ np.linalg.inv(LLS + alpha * noise_cov)
            W_dSPM = np.diag( np.sqrt( 1 / np.diagonal(K @ noise_cov @ K.T) ) )
            inverse_operator = W_dSPM @ K
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
  
class SolverMinimumL1Norm(BaseSolver):
    ''' Class for the Minimum Current Estimate (MCE) inverse solution using the
        FISTA solver [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging sciences,
    2(1), 183-202.
    '''
    def __init__(self, name="Minimum Current Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha='auto', max_iter=1000, noise_cov=None, verbose=0, **kwargs):
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
        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)

        self.noise_cov = noise_cov
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj, max_iter=1000, 
                               l1_reg=1e-3, l2_reg=0, tol=1e-2) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        max_iter : int
            Maximum number of iterations
        l1_reg : float
            Controls the spatial L1 regularization 
        l2_reg : float
            Controls the spatial L2 regularization 
        tol : float
            Tolerance at which convergence is met.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.fista_wrap(data, max_iter=max_iter, 
                                    l1_reg=l1_reg, l2_reg=l2_reg, tol=tol)
        stc = self.source_to_object(source_mat)
        return stc
    
    def fista_wrap(self, y_mat, max_iter=1000, l1_reg=1e-3, l2_reg=0, tol=1e-2):
        srcs = []
        for y in y_mat.T:
            srcs.append ( self.fista(y, max_iter=max_iter, 
                                    l1_reg=l1_reg, l2_reg=l2_reg, 
                                    tol=tol) 
                                    )
        return np.stack(srcs, axis=1)


    def fista(self, y, l1_reg=1e-3, l2_reg=0, max_iter=1000, tol=1e-2):
        """
        Solves the EEG inverse problem:
            min_x ||y - Ax||_2^2 + l1_reg * ||x||_1 + l2_reg * ||x||_2^2
        using the FISTA algorithm.
        
        Parameters
        ----------
        y : ndarray, shape (m,)
            EEG measurements.
        A : ndarray, shape (m, n)
            Forward model.
        x0 : ndarray, shape (n,)
            Initial guess for the CSDs.
        l1_reg : float, optional (default: 1e-3)
            L1 regularization strength.
        l2_reg : float, optional (default: 0)
            L2 regularization strength.
        max_iter : int, optional (default: 1000)
            Maximum number of iterations to run.
        tol : float, optional (default: 1e-6)
            Tolerance for the stopping criteria.
        
        Returns
        -------
        x : ndarray, shape (n,)
            Estimated CSDs.
        """
        
        def grad_f(x):
            """Gradient of the objective function"""
            return x - y + l1_reg * np.sign(x) + l2_reg * x

        A = deepcopy(self.leadfield)
        A /= np.linalg.norm(A, axis=0)

        y_scaled = y.copy()
        # Rereference
        y_scaled -= y_scaled.mean()
        # Scale to unit norm
        norm_y = np.linalg.norm(y_scaled)
        y_scaled /= norm_y
        
        # Calculate initial guess
        x0 = np.linalg.pinv(A) @ y_scaled
        # Scale to unit norm
        x0 /= np.linalg.norm(x0)

        x = x0.copy()
        y = x0.copy()
        
        t = 1.0
        lr = 1.0
        for i in range(max_iter):
            x_prev = x.copy()
            # Gradient descent step
            x = y - lr * grad_f(y)
            # Soft thresholding step
            x = self.soft_threshold(x, l1_reg * lr)
            # Update y and t
            t_prev = t
            t = (1 + (1 + 4 * t**2)**0.5) / 2
            y = x + (t_prev - 1) / t * (x - x_prev)

            if np.linalg.norm(y) == 0:
                break
            # Check stopping criteria
            if np.linalg.norm(x - x_prev) < tol:
                break
        # Rescale source
        x = x * norm_y

        return x
        

    @staticmethod
    def soft_threshold(x, alpha):
        """
        Applies the soft thresholding operator to x with threshold alpha.
        
        Parameters
        ----------
        x : ndarray, shape (n,)
            Input array.
        alpha : float
            Threshold.
        
        Returns
        -------
        y : ndarray, shape (n,)
            Output array.
        """
        y = np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
        return y


class SolverGFTMinimumL1Norm(BaseSolver):
    ''' Class for the Minimum Current Estimate (MCE) inverse solution using the
        FISTA solver [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging sciences,
    2(1), 183-202.
    '''
    def __init__(self, name="Minimum Current Estimate", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha='auto', max_iter=1000, noise_cov=None, verbose=0, **kwargs):
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
        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        
        adjacency = mne.spatial_src_adjacency(forward['src'], verbose=0)
        lap = laplacian(adjacency)
        # U, eigenvalues , _ = np.linalg.svd(lap.toarray(), full_matrices=False)
        eigenvalues, U = np.linalg.eig(lap.toarray())
        cutoff_index = int(len(eigenvalues) * 0.3)  # 30% cutoff
        U = U[:, -cutoff_index:]
        self.U = np.real(U)

        self.noise_cov = noise_cov
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj, max_iter=1000, 
                               l1_reg=1e-3, l2_reg=0, tol=1e-2) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        max_iter : int
            Maximum number of iterations
        l1_reg : float
            Controls the spatial L1 regularization 
        l2_reg : float
            Controls the spatial L2 regularization 
        tol : float
            Tolerance at which convergence is met.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)
        source_mat = self.fista_wrap(data, max_iter=max_iter, 
                                    l1_reg=l1_reg, l2_reg=l2_reg, tol=tol)
        stc = self.source_to_object(source_mat)
        return stc
    
    def fista_wrap(self, y_mat, max_iter=1000, l1_reg=1e-3, l2_reg=0, tol=1e-2):
        srcs = []
        for y in y_mat.T:
            srcs.append ( self.fista(y, max_iter=max_iter, 
                                    l1_reg=l1_reg, l2_reg=l2_reg, 
                                    tol=tol) 
                                    )
        return np.stack(srcs, axis=1)


    def fista(self, y, l1_reg=1e-3, l2_reg=0, max_iter=1000, tol=1e-2):
        """
        Solves the EEG inverse problem:
            min_x ||y - Ax||_2^2 + l1_reg * ||x||_1 + l2_reg * ||x||_2^2
        using the FISTA algorithm.
        
        Parameters
        ----------
        y : ndarray, shape (m,)
            EEG measurements.
        A : ndarray, shape (m, n)
            Forward model.
        x0 : ndarray, shape (n,)
            Initial guess for the CSDs.
        l1_reg : float, optional (default: 1e-3)
            L1 regularization strength.
        l2_reg : float, optional (default: 0)
            L2 regularization strength.
        max_iter : int, optional (default: 1000)
            Maximum number of iterations to run.
        tol : float, optional (default: 1e-6)
            Tolerance for the stopping criteria.
        
        Returns
        -------
        x : ndarray, shape (n,)
            Estimated CSDs.
        """
        
        def grad_f(x):
            """Gradient of the objective function"""
            return x - y + l1_reg * np.sign(x) + l2_reg * x

        A = deepcopy(self.leadfield) @ self.U
        # A /= np.linalg.norm(A, axis=0)

        y_scaled = y.copy()
        # Rereference
        y_scaled -= y_scaled.mean()
        # Scale to unit norm
        norm_y = np.linalg.norm(y_scaled)
        y_scaled /= norm_y
        
        # Calculate initial guess
        x0 = np.linalg.pinv(A) @ y_scaled
        # Scale to unit norm
        x0 /= np.linalg.norm(x0)

        x = x0.copy()
        y = x0.copy()
        
        t = 1.0
        lr = 1.0
        for i in range(max_iter):
            x_prev = x.copy()
            # Gradient descent step
            x = y - lr * grad_f(y)
            # Soft thresholding step
            x = self.soft_threshold(x, l1_reg * lr)
            # Update y and t
            t_prev = t
            t = (1 + (1 + 4 * t**2)**0.5) / 2
            y = x + (t_prev - 1) / t * (x - x_prev)

            if np.linalg.norm(y) == 0:
                print("norm is zero")
                x = x_prev
                break
            # Check stopping criteria
            # print(np.linalg.norm(x - x_prev))
            if np.linalg.norm(x - x_prev) < tol:
                print("criterion met")
                break
        # Rescale source
        x = x * norm_y

        return self.U @ x
        

    @staticmethod
    def soft_threshold(x, alpha):
        """
        Applies the soft thresholding operator to x with threshold alpha.
        
        Parameters
        ----------
        x : ndarray, shape (n,)
            Input array.
        alpha : float
            Threshold.
        
        Returns
        -------
        y : ndarray, shape (n,)
            Output array.
        """
        y = np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
        return y

class SolverMinimumL1NormGPT(BaseSolver):
    ''' Class for the Minimum Current Estimate inverse solution using
        interesting code from the Chat GPT AI by openai.com (GPT-solver). 
        
        I (Lukas Hecker) prompted the task to write a sparsified eLORETA-type
        inverse solution and this came up with little adjustments required.

        I can't express how weird it is for me, too.
    
    Attributes
    ----------
    
    References
    ----------
    Open AI chat GPT (openai.com)
    
    '''
    def __init__(self, name="GPT Solver", **kwargs):
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
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj, max_iter=1000, 
                               l1_reg=1e-3, tol=1e-2) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        max_iter : int
            Maximum number of iterations
        l1_reg : float
            Controls the spatial L1 regularization 
        tol : float
            Tolerance at which convergence is met.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.solver_wrap(data, max_iter=max_iter, 
                                    l1_reg=l1_reg, tol=tol)
        stc = self.source_to_object(source_mat)
        return stc
    
    def solver_wrap(self, y_mat, max_iter=1000, l1_reg=1e-3, tol=1e-2):
        srcs = []
        for y in y_mat.T:
            srcs.append ( self.solve(y, max_iter=max_iter, 
                                    l1_reg=l1_reg,
                                    tol=tol) 
                                    )
        return np.stack(srcs, axis=1)


    def solve(self, y, l1_reg=1e-3, max_iter=1000, tol=1e-2):
        """
        Solves the EEG inverse problem:
            min_x ||y - Ax||_2^2 + l1_reg * ||x||_1 + l2_reg * ||x||_2^2
        using the FISTA algorithm.
        
        Parameters
        ----------
        y : ndarray, shape (m,)
            EEG measurements.
        l1_reg : float, optional (default: 1e-3)
            L1 regularization strength.
        max_iter : int, optional (default: 1000)
            Maximum number of iterations to run.
        tol : float, optional (default: 1e-6)
            Tolerance for the stopping criteria.
        
        Returns
        -------
        x : ndarray, shape (n,)
            Estimated CSDs.
        """
        
        
        A = deepcopy(self.leadfield)
        A /= np.linalg.norm(A, axis=0)

        y_scaled = y.copy()
        # Rereference
        y_scaled -= y_scaled.mean()
        # Scale to unit norm
        norm_y = np.linalg.norm(y_scaled)
        y_scaled /= norm_y
        
        # Calculate initial guess
        x0 = np.linalg.pinv(A) @ y_scaled
        # Scale to unit norm
        x0 /= np.linalg.norm(x0)

        x = x0.copy()
        y = x0.copy()
        
        lr = 1.0
        for i in range(max_iter):
            x_prev = x.copy()
            # Gradient descent step
            x = x - lr*l1_reg * self.grad_f(x, A, y_scaled)
            # Soft thresholding step
            x = self.soft_threshold(x, lr*l1_reg)
            
            # Check stopping criteria
            if np.linalg.norm(x) == 0:
                x = x_prev
                break
            if np.linalg.norm(x - x_prev) < tol:
                break
        # Rescale source
        x = x * norm_y

        return x

    @staticmethod        
    def grad_f(x, A, y_scaled):
        """Gradient of the objective function"""
        return A.T.dot(A.dot(x) - y_scaled) + np.sign(x)

    @staticmethod
    def soft_threshold(x, alpha):
        """
        Applies the soft thresholding operator to x with threshold alpha.
        
        Parameters
        ----------
        x : ndarray, shape (n,)
            Input array.
        alpha : float
            Threshold.
        
        Returns
        -------
        y : ndarray, shape (n,)
            Output array.
        """
        y = np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
        return y

class SolverMinimumL1L2Norm(BaseSolver):
    ''' Class for the Minimum L1-L2 Norm solution (MCE) inverse solution. It
        imposes a L1 norm on the source and L2 on the source time courses.
    
    References
    ----------
    [!] Missing reference - please contact developers if you have it!

    '''
    def __init__(self, name="Minimum L1-L2 Norm", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha=0.01, **kwargs):
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
        
        return self

    def apply_inverse_operator(self, mne_obj, alpha="auto", 
                               max_iter=100, l1_spatial=1e-3, 
                               l2_temporal=1e-3, tol=1e-6, ) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        max_iter : int
            Maximum number of iterations
        l1_spatial : float
            Controls the spatial L1 regularization 
        l2_temporal : float
            Controls the temporal L2 regularization 
        tol : float
            Tolerance at which convergence is met.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.fista_eeg(data, alpha=alpha, max_iter=max_iter, 
                                    tol=tol, l1_spatial=l1_spatial, 
                                    l2_temporal=l2_temporal)
        stc = self.source_to_object(source_mat)
        return stc
    
    def fista_eeg(self, y, alpha="auto", l1_spatial=1e-3, l2_temporal=1e-3, 
                  max_iter=1000, tol=1e-6):
        """
        Solves the EEG inverse problem using FISTA with L1 regularization on the spatial
        dimension and L2 regularization on the temporal dimension.
        
        Parameters:
        - A: array of shape (n_sensors, n_sources)
        - y: array of shape (n_sensors, n_timepoints)
        - l1_spatial: float, strength of L1 regularization on the spatial dimension
        - l2_temporal: float, strength of L2 regularization on the temporal dimension
        - max_iter: int, maximum number of iterations
        - tol: float, tolerance for convergence
        
        Returns:
        - x: array of shape (n_sources, n_timepoints), the solution to the EEG inverse problem
        """
        A = self.leadfield
        A -= A.mean(axis=0)
        A /= np.linalg.norm(A, axis=0)

        norm_y = np.linalg.norm(y)
        y -= y.mean(axis=0)
        y_scaled = y / norm_y

        # Regularization
        if alpha == "auto":
            alpha = l1_spatial

        # Initialize x and z to be the same, and set t to 1
        W = np.diag(np.linalg.norm(A, axis=0))
        WTW = np.linalg.inv(W.T @ W)
        LWTWL = A @ WTW @ A.T
        inverse_operator = WTW @ A.T  @ np.linalg.inv(LWTWL + alpha * np.identity(A.shape[0]))
        x = z = inverse_operator @ y_scaled
        
        # x = z = np.linalg.pinv(A) @ y_scaled
        
        x /= np.linalg.norm(x)

        t = 1
        
        # Compute the Lipschitz constant
        L = np.linalg.norm(A, ord=2) ** 2
        
        for i in range(max_iter):
            # Compute the gradient of the smooth part of the objective
            grad = A.T @ (A @ x - y_scaled)
            
            # Compute the proximal operator of the L1 regularization
            x_new = np.sign(x - grad / L) * np.maximum(np.abs(x - grad / L) - l1_spatial / L, 0)
            
            # Compute the proximal operator of the L2 regularization
            x_new = x_new - np.mean(x_new, axis=0)
            x_new = x_new / np.linalg.norm(x_new, ord=2, axis=0) * np.maximum(np.linalg.norm(x_new, ord=2, axis=0) - l2_temporal / L, 0)
            x_new = x_new + np.mean(x_new, axis=0)
            
            # Update t and z
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z_new = x_new + (t - 1) / t_new * (x_new - x)
            
            # Check for convergence
            if np.linalg.norm(x_new - x) < tol:
                # print("convergence after ", i)
                break
                
            # Update x, t, and z
            x = x_new
            t = t_new
            z = z_new
        # Rescale Sources
        x = x * norm_y

        return x


    @staticmethod
    def calc_norm(x, n_time):
        return np.sqrt( (x**2).sum() / n_time )