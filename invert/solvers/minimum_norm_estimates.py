import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator

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
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        leadfield = self.leadfield
        n_chans, _ = leadfield.shape
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = leadfield.T @ np.linalg.inv(leadfield @ leadfield.T + alpha * np.identity(n_chans))
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)


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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)
  
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

    def apply_inverse_operator(self, evoked, max_iter=1000, 
                               l1_reg=1e-3, l2_reg=0, tol=1e-2) -> mne.SourceEstimate:

        source_mat = self.fista_wrap(evoked.data, max_iter=max_iter, 
                                    l1_reg=1e-3, l2_reg=0, tol=1e-2)
        stc = self.source_to_object(source_mat, evoked)
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
            first = y-x
            second = l1_reg * np.sign(x) 
            third = l2_reg * x
            # print(first.shape, second.shape, third.shape)
            return first + second + third
            # return x - y + l1_reg * np.sign(x) + l2_reg * x

        A = deepcopy(self.leadfield)
        A /= np.linalg.norm(A, axis=0)

        y_scaled = y.copy()
        # Rereference
        y_scaled -= y_scaled.mean()
        # Scale to unit norm
        y_scaled /= np.linalg.norm(y_scaled)
        
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

    def apply_inverse_operator(self, evoked, max_iter=1000, 
                               l1_reg=1e-3, tol=1e-2) -> mne.SourceEstimate:

        source_mat = self.solver_wrap(evoked.data, max_iter=max_iter, 
                                    l1_reg=l1_reg, tol=tol)
        stc = self.source_to_object(source_mat, evoked)
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
        y_scaled /= np.linalg.norm(y_scaled)
        
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
            # print(np.linalg.norm(x - x_prev))
            if np.linalg.norm(x) == 0:
                x = x_prev
                break
            if np.linalg.norm(x - x_prev) < tol:
                break
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


# class SolverMinimumL1Norm(BaseSolver):
#     ''' Class for the Minimum Current Estimate (MCE) inverse solution using the
#         FISTA solver [1].
    
#     Attributes
#     ----------
    
#     References
#     ----------
#     [1] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
#     algorithm for linear inverse problems. SIAM journal on imaging sciences,
#     2(1), 183-202.
#     '''
#     def __init__(self, name="Minimum Current Estimate", **kwargs):
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
        
#         super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
#         n_chans = self.leadfield.shape[0]
#         if noise_cov is None:
#             noise_cov = np.identity(n_chans)

#         self.noise_cov = noise_cov
#         self.inverse_operators = []
#         return self

#     def apply_inverse_operator(self, evoked, max_iter=1000) -> mne.SourceEstimate:

#         source_mat = self.fista_wrap(evoked.data, max_iter=max_iter)
#         stc = self.source_to_object(source_mat, evoked)
#         return stc
    
#     def fista_wrap(self, y_mat, max_iter=1000):
#         srcs = []
#         for y in y_mat.T:
#             srcs.append ( self.fista(y, max_iter=max_iter) )
#         return np.stack(srcs, axis=1)


#     def fista(self, y, max_iter=1000):
#         ''' The FISTA algorithm based on [1].

#         Parameters 
#         ---------- 
#         y : numpy.ndarray
#             The observations (i.e., eeg/meg data at single time point)
#         max_iter : int
#             Maximum number of iterations for the FISTA algorithm.
        
#         References
#         ----------
#         [1] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
#         algorithm for linear inverse problems. SIAM journal on imaging sciences,
#         2(1), 183-202.
#         '''
#         _, n_dipoles = self.leadfield.shape
#         beta = 1 / np.sum(self.leadfield**2)
#         lam = 1e-14
#         patience = 1000
#         x_t = np.zeros(n_dipoles)
#         x_t_prev = np.zeros(n_dipoles)
#         x_best = np.zeros(n_dipoles)
#         error_best = np.inf
#         errors = []
#         A_H = np.matrix(self.leadfield).getH()
#         for t in range(max_iter):
#             v_t = y - self.leadfield @ x_t
            
#             r = x_t + beta * A_H @ v_t + ((t-2)/(t+1)) * (x_t - x_t_prev)
#             print(x_t.shape, v_t.shape, A_H.shape)
#             x_tplus = self.soft_thresholding(r, lam)
            
#             x_t_prev = deepcopy(x_t)
#             x_t = x_tplus
#             error = np.sum((y - self.leadfield @ x_t)**2) * 0.5 + lam * abs(x_t).sum()
#             errors.append( error )

#             if errors[-1] < error_best:
#                 x_best = deepcopy(x_t)
#                 error_best = errors[-1]
#             if t>patience and  (np.any(np.isnan(x_tplus))  or np.all(np.array(errors[-patience:-1]) < errors[-1] )):
#                 print("breaking")
#                 break

#             if self.verbose>0:
#                 if np.mod(t, 1000) == 0:
#                     print(f"iter {t} error {errors[-1]} maxval {abs(x_t).max()}")
#         if self.verbose>0:
#             print(f"Finished after {t} iterations, error = {error_best}")
#         return x_best
        

#     @staticmethod
#     def soft_thresholding(r, lam):
#         r = np.squeeze(np.array(r))
#         C = np.sign(r) * np.clip(abs(r) - lam, a_min=0, a_max=None)
#         return C

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

    def apply_inverse_operator(self, evoked, max_iter=100, min_change=0.005) -> mne.SourceEstimate:
        ''' Apply the L1L2 inverse operator.
        Parameters
        ----------
        evoked : mne.Evoked
            The mne M/EEG data object.
        max_iter : int
            Maximum number of iterations (stopping criterion).
        min_change : float
            Convergence criterion.

        Return
        ------
        stc : mne.SourceEstimate
            The source estimate object.
        '''
        source_mat = self.calc_l1l2_solution(evoked.data, max_iter=max_iter, min_change=min_change)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    
    def calc_l1l2_solution(self, y, max_iter=100, min_change=0.005):
        ''' Calculate the L1L2 inverse solution.
        Parameters
        ----------
        y : numpy.ndarray
            The M/EEG data matrix.
        max_iter : int
            Maximum number of iterations (stopping criterion).
        min_change : float
            Convergence criterion.

        Return
        ------
        x_hat :numpy.ndarray
            The inverse solution matrix.
        '''

        leadfield = self.leadfield
        _, n_dipoles = leadfield.shape
        n_chans, n_time = y.shape

        if self.alpha == "auto":
            _, s, _ = np.linalg.svd(leadfield)
            self.alpha = 1e-7 # * s.max()
        eps = 1e-16
        leadfield -= leadfield.mean(axis=0)
        y -= y.mean(axis=0)
        I = np.identity(n_chans)
        x_hat = np.ones((n_dipoles, n_time))

        LLT = [ leadfield[:, rr][:, np.newaxis] @ leadfield[:, rr][:, np.newaxis].T for rr in range(n_dipoles)]
        L1_norms = [1e99,]

        for i in range(max_iter):
            y_hat = leadfield @ x_hat
            y_hat -= y_hat.mean(axis=0)
            # R = np.linalg.norm(y - y_hat)
            # print(i, " Residual: ", R)
            norms = [self.calc_norm(x_hat[rr, :], n_time) for rr in range(n_dipoles)]
            ALLT = np.stack( [ norms[rr] * LLT[rr]  for rr in range(n_dipoles)], axis=0).sum(axis=0)
            for r in range(n_dipoles):
                Lr = leadfield[:, r][:, np.newaxis]
                x_hat[r, :] = norms[r] * Lr.T @ np.linalg.inv( ALLT + self.alpha * I ) @ y
            L1_norms.append( np.abs(x_hat).sum() )
            current_change = 1 - L1_norms[-1] / (L1_norms[-2]+eps)
            # current_change = L1_norms[-2] - L1_norms[-1]
            print(current_change)
            if current_change < min_change:
                # print(f"Percentage change is {100*current_change:.4f} % (below {100*(min_change):.1f} %) - stopping")
                break
        return x_hat


    @staticmethod
    def calc_norm(x, n_time):
        return np.sqrt( (x**2).sum() / n_time )
        
