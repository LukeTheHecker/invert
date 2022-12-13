import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator
from ..util import calc_residual_variance, thresholding, find_corner, best_index_residual

class SolverOMP(BaseSolver):
    ''' Class for the Orthogonal Matching Pursuit (OMP) inverse solution [1].
        The algorithm as described by [2] was implemented.
    
    Attributes
    ----------
    
    References
    ----------
    [1] Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random
    measurements via orthogonal matching pursuit. IEEE Transactions on
    information theory, 53(12), 4655-4666. 
    
    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    '''
    def __init__(self, name="Orthogonal Matching Pursuit", **kwargs):
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
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield,axis=0)
        
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, K=1) -> mne.SourceEstimate:
        source_mat = np.stack([self.calc_omp_solution(y, K=K) for y in evoked.data.T], axis=1)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_omp_solution(self, y, K=1):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        K : int
            The number of atoms to select per iteration.
        
        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        y -= y.mean()
        
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape
        # leadfield_pinv = np.linalg.pinv(self.leadfield)
        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        # unexplained_variance = np.array([calc_residual_variance(y, leadfield@x_hat),])
        # source_norms = np.array([0,])

        omega = np.array([])
        r = deepcopy(y)
        y_hat = self.leadfield@x_hat
        y_hat -= y_hat.mean()
        residuals = np.array([np.linalg.norm(y - y_hat), ])

        for _ in range(n_chans):
            # b = self.leadfield.T @ r
            b = self.leadfield_normed.T @ r

            b_thresh = thresholding(b, K)
            omega = np.append(omega, np.where(b_thresh!=0)[0])  # non-zero idc
            omega = omega.astype(int)

            x_hat[omega] = np.linalg.pinv(self.leadfield[:, omega]) @ y

            y_hat = self.leadfield@x_hat
            y_hat -= y_hat.mean()
            r = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            # source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))


            
        # iters = np.arange(len(residuals)).astype(float)
        # corner_idx = find_corner(iters, residuals)
        # x_hat = x_hats[corner_idx]

        x_hat = best_index_residual(residuals, x_hats, plot=False)
        
        return x_hat

class SolverSOMP(BaseSolver):
    ''' Class for the Simultaneous Orthogonal Matching Pursuit (S-OMP) inverse
        solution.
    
    Attributes
    ----------

    
    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.
    
    [2] Donoho, D. L. (2006). Compressed sensing. IEEE Transactions on
    information theory, 52(4), 1289-1306.

    '''
    def __init__(self, name="Simultaneous Orthogonal Matching Pursuit", **kwargs):
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
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield,axis=0)
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, K=1) -> mne.SourceEstimate:
        source_mat = self.calc_somp_solution(evoked.data, K)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_somp_solution(self, y, K):
        """ Calculates the S-OMP inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, time).
        K : int
            The number of atoms to select per iteration.
        
        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, time)

        """
        n_chans, n_time = y.shape
        max_iter = int(n_chans/2)
        _, n_dipoles = self.leadfield.shape

        leadfield_pinv = np.linalg.pinv(self.leadfield)
        y -= y.mean(axis=0)

        x_hat = np.zeros((n_dipoles, n_time))
        x_hats = [deepcopy(x_hat)]
        residuals = np.array([np.linalg.norm(y - self.leadfield@x_hat), ])
        # unexplained_variance = np.array([calc_residual_variance(y, self.leadfield@x_hat),])
        source_norms = np.array([0,])

        R = deepcopy(y)
        omega = np.array([])
        q = 1
        for i in range(max_iter):
            # b_n = np.linalg.norm(self.leadfield.T@R, axis=1, ord=q)
            b_n = np.linalg.norm(self.leadfield_normed.T@R, axis=1, ord=q)

            # if len(omega)>0:
            #     b_n[omega] = 0

            b_thresh = thresholding(b_n, K)
            omega = np.append(omega, np.where(b_thresh!=0)[0])  # non-zero idc
            omega = np.unique(omega.astype(int))
            leadfield_pinv = np.linalg.pinv(self.leadfield[:, omega])
            x_hat[omega] = leadfield_pinv @ y
            
            y_hat = self.leadfield@x_hat
            y_hat -= y_hat.mean(axis=0)

            R = y - y_hat
            
            residuals = np.append(residuals, np.linalg.norm(R))
            # unexplained_variance = np.append(unexplained_variance, calc_residual_variance(y, self.leadfield@x_hat))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append( deepcopy(x_hat) )
        
        # unexplained_variance[0] = unexplained_variance[1]
        # iters = np.arange(len(residuals))
        # corner_idx = find_corner(residuals, iters)
        # x_hat = x_hats[corner_idx]

        x_hat = best_index_residual(residuals, x_hats, plot=False)
        
        return x_hat

class SolverCOSAMP(BaseSolver):
    ''' Class for the Compressed Sampling Matching Pursuit (CoSaMP) inverse
        solution [1]. The algorithm as described in [2] was used for this
        imlementation.
    
    Attributes
    ----------

    
    References
    ----------
    [1] Needell, D., & Tropp, J. A. (2009). CoSaMP: Iterative signal recovery
    from incomplete and inaccurate samples. Applied and computational harmonic
    analysis, 26(3), 301-321. 
    
    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.
    '''
    def __init__(self, name="Compressed Sampling Matching Pursuit", **kwargs):
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
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield,axis=0)
  
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, K="auto", rv_thresh=1) -> mne.SourceEstimate:
        source_mat = np.stack([self.calc_cosamp_solution(y, K=K,  rv_thresh=rv_thresh) for y in evoked.data.T], axis=1)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_cosamp_solution(self, y, K='auto', rv_thresh=1):
        """ Calculates the CoSaMP inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels, time).
        K : int
            Positive integer determining the sparsity of the reconstructed
            signal.
        rv_thresh : float
            The residual variance threshold as a stopping criterion. The
            smaller, the sooner the atom search is considered complete, i.e.,
            the less of the data is fitted. It can therefore be used for
            regularization.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles, time)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        if K == "auto":
            K = 3
        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        b = np.zeros((n_dipoles, ))
        r = deepcopy(y)
        y -= y.mean()
        self.leadfield -= self.leadfield.mean(axis=0)
        y_hat = self.leadfield@x_hat
        y_hat -= y_hat.mean()
        residuals = np.array([np.linalg.norm(y - y_hat), ])
        source_norms = np.array([0,])
        unexplained_variance = np.array([calc_residual_variance(self.leadfield@x_hat, y),])

        for i in range(1, n_chans+1):
            # e = self.leadfield.T @ r
            e = self.leadfield_normed.T @ r
            e_thresh = thresholding(e, 2*K)
            omega = np.where(e_thresh!=0)[0]
            old_activations = np.where(x_hats[i-1]!=0)[0]
            T = np.unique(np.concatenate([omega, old_activations]))
            # print(len(T))
            leadfield_pinv = np.linalg.pinv(self.leadfield[:, T])
            # leadfield_pinv = np.linalg.pinv(self.leadfield)[T]
            
            b[T] = leadfield_pinv @ y
            x_hat = thresholding(b, 2*K-1)
            y_hat = self.leadfield@x_hat
            y_hat -= y_hat.mean()
            r = y - y_hat
            r -= r.mean(axis=0)
            
            residuals = np.append(residuals, np.linalg.norm(r))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            unexplained_variance = np.append(unexplained_variance, calc_residual_variance(self.leadfield@x_hat, y))
            x_hats.append(deepcopy(x_hat))
            # if :
            #     break
            if residuals[-1] >= residuals[-2] or unexplained_variance[-1] < rv_thresh:
                break 
        
        
        # iters = np.arange(len(residuals))
        # corner_idx = find_corner(residuals, iters)
        # x_hat = x_hats[corner_idx]
        
        x_hat = best_index_residual(residuals, x_hats)
        
        return x_hat
        


class SolverREMBO(BaseSolver):
    ''' Class for the Reduce Multi-Measurement-Vector and Boost (ReMBo) inverse
        solution [1]. The algorithm as describe in [2] was used for the
        imlpementation.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Mishali, M., & Eldar, Y. C. (2008). Reduce and boost: Recovering
    arbitrary sets of jointly sparse vectors. IEEE Transactions on Signal
    Processing, 56(10), 4692-4702. 
    [2] Duarte, M. F., & Eldar, Y. C. (2011).
    Structured compressed sensing: From theory to applications. IEEE
    Transactions on signal processing, 59(9), 4053-4085.  
    '''
    def __init__(self, name="Reduce Multi-Measurement-Vector and Boost", **kwargs):
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

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, K=1) -> mne.SourceEstimate:
        """ Apply the REMBO inverse solution.
        
        Parameters
        ----------
        evoked : mne.Evoked
            The evoked object holding the M/EEG data.
        K : int
            The number of atoms to select per iteration.
        
        Return
        ------
        stc : mne.SourceEstimate
            The source estimate containing the inverse solution.
        """
        source_mat = self.calc_rembo_solution(evoked.data, K=K)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    
    def calc_rembo_solution(self, y, K=1):
        """ Calculate the REMBO inverse solution based on the measurement vector
            y.

        Parameters
        ----------
        y : numpy.ndarray
            The EEG matrix (channels, time)
        K : int
            The number of atoms to select per iteration.

        Return
        ------
        x_hat : numpy.ndarray
            The source matrix (dipoles, time)
        """

        rand = np.random.rand
        n_chans, n_time = y.shape
        max_iter = int(n_chans/2)
        n_dipoles = self.leadfield.shape[1]

        unexplained_variance = np.array([calc_residual_variance(np.zeros(y.shape), y),])
        n_sources = np.array([0,])
        x_hats = [np.zeros((n_dipoles, n_time))]

        for i in range(max_iter):
            a = rand(n_time)
            y_vec = y@a  # sample randomly from the measurement matrix
            x_hat = self.calc_omp_solution(y_vec, K=K)
            S_hat = np.where(x_hat!=0)[0].astype(int)
            As_pinv = np.linalg.pinv(self.leadfield[:, S_hat])

            x_hat = np.zeros((n_dipoles, n_time))
            x_hat[S_hat, :] = As_pinv @ y
            x_hats.append( x_hat )
            
            unexplained_variance = np.append(unexplained_variance, calc_residual_variance(self.leadfield@x_hat, y))
            n_sources = np.append(n_sources, len(S_hat)).astype(float)


        idc = np.argsort(n_sources)
        n_sources = n_sources[idc]
        unexplained_variance = unexplained_variance[idc]

        # REFACTOR THIS:
        unique_sources = np.sort(np.unique(n_sources))
        n_sources_new = []
        unexp_new = []
        x_hats_new = []
        for k in unique_sources:
            idc = np.where(n_sources == k)[0]
            values = unexplained_variance[idc]
            min_sidx = np.argmin(values)
            good_index = idc[min_sidx]

            n_sources_new.append(k)
            unexp_new.append(values.min())
            x_hats_new.append( x_hats[good_index] )
        n_sources = np.array(n_sources_new)[1:]
        unexplained_variance = np.array(unexp_new)[1:]
        x_hats = x_hats_new[1:]
        corner_idx = find_corner(n_sources, unexplained_variance)
        
        # corner_idx = find_corner(n_sources[1:], unexplained_variance[1:])+1
        # corner_idx = idc[corner_idx]
        x_hat = x_hats[corner_idx]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(n_sources, unexplained_variance, 'k*')
        plt.plot(n_sources[corner_idx], unexplained_variance[corner_idx], 'ro')
        
        
        return x_hat

    def calc_omp_solution(self, y, K=1):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        K : int
            The number of atoms to select per iteration.
        
        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        # leadfield_pinv = np.linalg.pinv(self.leadfield)
        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        # unexplained_variance = np.array([calc_residual_variance(y, leadfield@x_hat),])
        source_norms = np.array([0,])

        x_hat = np.zeros((n_dipoles, ))
        omega = np.array([])
        r = deepcopy(y)
        residuals = np.array([np.linalg.norm(y - self.leadfield@x_hat), ])
        source_norms = np.array([0,])
        x_hats = [deepcopy(x_hat), ]

        for i in range(n_chans):
            b = self.leadfield.T @ r
            b_thresh = thresholding(b, K)
            omega = np.append(omega, np.where(b_thresh!=0)[0])  # non-zero idc
            omega = omega.astype(int)

            x_hat[omega] = np.linalg.pinv(self.leadfield[:, omega]) @ y
            r = y - self.leadfield@x_hat

            residuals = np.append(residuals, np.linalg.norm(y - self.leadfield@x_hat))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))


            
        iters = np.arange(len(residuals)).astype(float)
        corner_idx = find_corner(iters, residuals)
        x_hat = x_hats[corner_idx]
        return x_hat


class SolverSP(BaseSolver):
    ''' Class for the Subspace Pursuit (SP) inverse solution [1]. The algorithm
        as described by [2] was implemented.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Dai, W., & Milenkovic, O. (2009). Subspace pursuit for compressive
    sensing signal reconstruction. IEEE transactions on Information Theory,
    55(5), 2230-2249.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085. 
    '''
    def __init__(self, name="Subspace Pursuit", **kwargs):
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
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield,axis=0)
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, K=1) -> mne.SourceEstimate:
        """ Apply the SP inverse solution.
        
        Parameters
        ----------
        evoked : mne.Evoked
            The evoked object holding the M/EEG data.
        K : int
            The number of atoms to select per iteration.
        
        Return
        ------
        stc : mne.SourceEstimate
            The source estimate containing the inverse solution.
        """

        source_mat = np.stack([self.calc_sp_solution(y, K=K) for y in evoked.data.T], axis=1)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_sp_solution(self, y, K="auto"):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        K : ["auto", int]
            The number of atoms to select per iteration.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape

        if K == "auto":
            K = int(n_chans/2)

        resid = lambda y, phi: y - phi@np.linalg.pinv(phi)@y
        y -= y.mean(axis=0)
        b = self.leadfield.T @ y
        T0 = np.where(thresholding(b, K) != 0)[0]
        # print(K, T0)
        # R = resid(y, self.leadfield_normed[:, T0])
        R = resid(y, self.leadfield[:, T0])
        
        R -= R.mean()
        T_list = [T0, ]
        R_list = [R, ]

        for i in range(1, n_chans+1):
            b = self.leadfield_normed.T @ R_list[-1]
            # b = self.leadfield.T @ R_list[-1]
            

            new_T = np.where(thresholding(b, K) != 0)[0]
            T_tilde = np.unique(np.concatenate([T_list[i-1], new_T]))
            
            xp = np.linalg.pinv(self.leadfield[:, T_tilde]) @ y
            T_l = T_tilde[np.where(thresholding(xp, K) != 0)[0]]
            T_list.append( T_l )
            R = resid(y, self.leadfield[:, T_l])
            R -= R.mean()
        
            R_list.append( R )
            # print(np.linalg.norm(R_list[-1]))
            if np.linalg.norm(R_list[-1]) > np.linalg.norm(R_list[-2]) or i==n_chans:
                T_l = T_list[-2]
            else:
                T_l = T_list[-1]
        x_hat = np.zeros(n_dipoles)
        x_hat[T_l] = np.linalg.pinv(self.leadfield[:, T_l]) @ y
        return x_hat


class SolverSSP(BaseSolver):
    ''' Class for the Simultaneous Subspace Pursuit (SSP) inverse solution.
    
    Attributes
    ----------

    
    References
    ----------
    [1] Dai, W., & Milenkovic, O. (2009). Subspace pursuit for
    compressive sensing signal reconstruction. IEEE transactions on Information
    Theory, 55(5), 2230-2249.

    [2] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085. 
    '''
    def __init__(self, name="Simultaneous Subspace Pursuit", **kwargs):
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
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield,axis=0)
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, K=1) -> mne.SourceEstimate:
        """ Apply the SSP inverse solution.
        
        Parameters
        ----------
        evoked : mne.Evoked
            The evoked object holding the M/EEG data.
        K : int
            The number of atoms to select per iteration.
        
        Return
        ------
        stc : mne.SourceEstimate
            The source estimate containing the inverse solution.
        """

        source_mat = self.calc_ssp_solution(evoked.data, K=K)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_ssp_solution(self, y, K="auto"):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        K : ["auto", int]
            The number of atoms to select per iteration.
            
        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans, n_time = y.shape
        _, n_dipoles = self.leadfield.shape

        if K == "auto":
            K = int(n_chans/2)

        resid = lambda y, phi: y - phi@np.linalg.pinv(phi)@y
        # resid = lambda y, phi: y - np.linalg.pinv(phi)@y
        
        y -= y.mean(axis=0)

        b = np.linalg.norm(self.leadfield.T @ y, axis=1)
        # b = np.linalg.norm(self.leadfield_normed.T @ y, axis=1)
        
        T0 = np.where(thresholding(b, K) != 0)[0]
        R = resid(y, self.leadfield[:, T0])
        
        R -= R.mean(axis=0)
        T_list = [T0, ]
        R_list = [R, ]

        for i in range(1, n_chans+1):
            b = np.linalg.norm(self.leadfield.T @ R, axis=1)
            # b = np.linalg.norm(self.leadfield_normed.T @ R, axis=1)
        

            new_T = np.where(thresholding(b, K) != 0)[0]
            T_tilde = np.unique(np.concatenate([T_list[i-1], new_T]))
            
            xp = np.linalg.pinv(self.leadfield[:, T_tilde]) @ y
            xp_norm = np.linalg.norm(xp, axis=1)

            T_l = T_tilde[np.where(thresholding(xp_norm, K) != 0)[0]]
            T_list.append( T_l )
            xp = np.linalg.pinv(self.leadfield[:, T_l]) @ y
            # R = resid(y, self.leadfield[:, T_l])
            R = y -  self.leadfield[:, T_l] @ xp
            R -= R.mean(axis=0)
        
            R_list.append( R )
            # print(np.linalg.norm(R_list[-1]))
            if np.linalg.norm(R_list[-1]) > np.linalg.norm(R_list[-2]) or i==n_chans:
                T_l = T_list[-2]
            else:
                T_l = T_list[-1]
        x_hat = np.zeros((n_dipoles, n_time))
        x_hat[T_l] = np.linalg.pinv(self.leadfield[:, T_l]) @ y
        return x_hat
