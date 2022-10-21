import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator
from scipy.sparse.csgraph import laplacian
from ..util import best_index_residual, thresholding, calc_residual_variance, find_corner

class SolverSMP(BaseSolver):
    ''' Class for the Smooth Matching Pursuit (SMP) inverse solution. Developed
        by Lukas Hecker as a smooth extension of the orthogonal matching pursuit
        algorithm [1], 19.10.2022.
    
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    '''
    def __init__(self, name="Smooth Matching Pursuit", **kwargs):
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
        self.leadfield = leadfield
        self.inverse_operators = []
        
        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0).toarray()
        laplace_operator = laplacian(adjacency)
        self.laplace_operator = laplace_operator
        leadfield_smooth = leadfield @ abs(laplace_operator)

        leadfield_smooth -= leadfield_smooth.mean(axis=0)
        self.leadfield -= self.leadfield.mean(axis=0)
        self.leadfield_smooth = leadfield_smooth
        self.leadfield_smooth_normed = self.leadfield_smooth / np.linalg.norm(self.leadfield_smooth, axis=0)
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield, axis=0)
        
        return self

    def apply_inverse_operator(self, evoked, K=1, include_singletons=True) -> mne.SourceEstimate:
        source_mat = np.stack([self.calc_smp_solution(y, include_singletons=include_singletons) for y in evoked.data.T], axis=1)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_smp_solution(self, y, include_singletons=True, residual_thresh=0.01):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        include_singletons : bool
            If True -> Include not only smooth patches but also single dipoles.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape
        
        y -= y.mean()
        x_hat = np.zeros(n_dipoles)
        omega = np.array([])
        r = deepcopy(y)
        y_hat = self.leadfield@x_hat
        y_hat -= y_hat.mean(axis=0)
        residuals = np.array([np.linalg.norm(y - y_hat), ])
        unexplained_variance = np.array([calc_residual_variance(y_hat, y),])
        # source_norms = np.array([0,])
        x_hats = [deepcopy(x_hat), ]

        for _ in range(n_chans):
            b_smooth = self.leadfield_smooth_normed.T @ r
            b_sparse = self.leadfield_normed.T @ r

            if include_singletons & (abs(b_sparse).max() > abs(b_smooth).max()):  # if sparse is better
                b_sparse_thresh = thresholding(b_sparse, 1)
                new_patch = np.where(b_sparse_thresh != 0)[0]
                
            else: # else if patch is better
                b_smooth_thresh = thresholding(b_smooth, 1)
                new_patch = np.where(self.laplace_operator[b_smooth_thresh!=0][0]!=0)[0]


            omega = np.append(omega, new_patch)
            omega = omega.astype(int)
            x_hat[omega] = np.linalg.pinv(self.leadfield[:, omega]) @ y
            y_hat = self.leadfield@x_hat
            y_hat -= y_hat.mean(axis=0)
            r = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            unexplained_variance = np.append(unexplained_variance, calc_residual_variance(y_hat, y))
            # source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))
            if residuals[-1] > residuals[-2]:
                break



        
        # Remove where residual is below threshold:
        # keep_idc = np.where(unexplained_variance>=residual_thresh*100)[0]
        # if len(keep_idc) == 1:
        #     keep_idc = np.append(keep_idc, 1)
        
        # print(keep_idc)
        # unexplained_variance = unexplained_variance[keep_idc]
        # x_hats = [x_hats[idx] for idx in keep_idc]
        
        x_hat = best_index_residual(unexplained_variance, x_hats, plot=False)
        
        # x_hat = x_hats[corner_idx]
        return x_hat

class SolverSSMP(BaseSolver):
    ''' Class for the Smooth Simultaneous Matching Pursuit (SSMP) inverse
        solution. Developed by Lukas Hecker as a smooth extension of the
        orthogonal matching pursuit algorithm [1], 19.10.2022.
    
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    '''
    def __init__(self, name="Smooth Simultaneous Matching Pursuit", **kwargs):
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
        self.leadfield = leadfield
        self.inverse_operators = []
        
        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0).toarray()
        laplace_operator = laplacian(adjacency)
        self.laplace_operator = laplace_operator
        leadfield_smooth = leadfield @ abs(laplace_operator)

        leadfield_smooth -= leadfield_smooth.mean(axis=0)
        self.leadfield -= self.leadfield.mean(axis=0)
        self.leadfield_smooth = leadfield_smooth
        self.leadfield_smooth_normed = self.leadfield_smooth / np.linalg.norm(self.leadfield_smooth, axis=0)
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield, axis=0)
        
        return self

    def apply_inverse_operator(self, evoked, K=1, include_singletons=True) -> mne.SourceEstimate:
        source_mat = self.calc_ssmp_solution(evoked.data, include_singletons=include_singletons)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    

    def calc_ssmp_solution(self, y, include_singletons=True):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        include_singletons : bool
            If True -> Include not only smooth patches but also single dipoles.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans, n_time = y.shape
        _, n_dipoles = self.leadfield.shape
        
        y -= y.mean(axis=0)
        x_hat = np.zeros(n_dipoles)
        x_hats = [deepcopy(x_hat)]
        source_norms = np.array([0,])

        x_hat = np.zeros((n_dipoles, n_time))
        omega = np.array([])
        R = deepcopy(y)
        y_hat = self.leadfield@x_hat
        y_hat -= y_hat.mean(axis=0)
        residuals = np.array([np.linalg.norm(y - y_hat), ])
        source_norms = np.array([0,])
        x_hats = [deepcopy(x_hat), ]
        q = 1
        for _ in range(n_chans):
            b_n_smooth = np.linalg.norm(self.leadfield_smooth_normed.T @ R, axis=1, ord=q)
            b_n_sparse = np.linalg.norm(self.leadfield_normed.T @ R, axis=1, ord=q)

            if include_singletons & (abs(b_n_sparse).max() > abs(b_n_smooth).max()):  # if sparse is better
                b_n_sparse_thresh = thresholding(b_n_sparse, 1)
                new_patch = np.where(b_n_sparse_thresh != 0)[0]
                
            else: # else if patch is better
                b_n_smooth_thresh = thresholding(b_n_smooth, 1)
                new_patch = np.where(self.laplace_operator[b_n_smooth_thresh!=0][0]!=0)[0]


            omega = np.append(omega, new_patch)
            omega = omega.astype(int)
            x_hat[omega] = np.linalg.pinv(self.leadfield[:, omega]) @ y
            
            y_hat = self.leadfield@x_hat
            y_hat -= y_hat.mean(axis=0)
            R = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(y - y_hat))
            source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))
            if residuals[-1] > residuals[-2]:
                break



        # Model selection (Regularisation)
        x_hat = best_index_residual(residuals, x_hats)
        # x_hat = x_hats[corner_idx]
        return x_hat
    
class SolverSubSMP(BaseSolver):
    ''' Class for the Subspace Smooth Matching Pursuit (SubSMP) inverse solution. Developed
        by Lukas Hecker as a smooth extension of the orthogonal matching pursuit
        algorithm [1], 19.10.2022.
    
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    
    References
    ----------
    [1] Duarte, M. F., & Eldar, Y. C. (2011). Structured compressed sensing:
    From theory to applications. IEEE Transactions on signal processing, 59(9),
    4053-4085.

    '''
    def __init__(self, name="Subspace Smooth Matching Pursuit", **kwargs):
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
        self.leadfield = leadfield
        self.inverse_operators = []
        
        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0).toarray()
        laplace_operator = laplacian(adjacency)
        self.laplace_operator = laplace_operator
        leadfield_smooth = leadfield @ abs(laplace_operator)

        leadfield_smooth -= leadfield_smooth.mean(axis=0)
        self.leadfield -= self.leadfield.mean(axis=0)
        self.leadfield_smooth = leadfield_smooth
        self.leadfield_smooth_normed = self.leadfield_smooth / np.linalg.norm(self.leadfield_smooth, axis=0)
        self.leadfield_normed = self.leadfield / np.linalg.norm(self.leadfield, axis=0)
        
        return self

    def apply_inverse_operator(self, evoked, K=1, include_singletons=True) -> mne.SourceEstimate:
        source_mat = self.calc_subsmp_solution(evoked.data, include_singletons=include_singletons)
        stc = self.source_to_object(source_mat, evoked)
        return stc

    def calc_subsmp_solution(self, y, include_singletons=True, ):
        ''' Calculate the Subspace Smooth Matching Pursuit (SubSMP) solution.
        '''
        _, n_time = y.shape
        n_dipoles = self.leadfield.shape[1]
        
        y -= y.mean(axis=0)
        u, s, _ = np.linalg.svd(y)
        
        # Select components
        idx = find_corner(np.arange(len(s)), s) + 0
        # Kaiser-Guttmann:
        # thr = 1  # 1=kaiser-guttmann, else: np.e**(-16)
        # idx = np.where((s*len(s) / s.sum()) > thr)[0][-1] + 1

        # Find matching sources to each of the selected eigenvectors (components)
        topos = u[:, :idx]
        x_hats = []
        for topo in topos.T:
            x_hats.append( self.calc_smp_solution(topo, include_singletons=include_singletons) )
        
        # Extract all active dipoles and patches:
        omegas = np.unique(np.where((np.stack(x_hats, axis=0)**2).sum(axis=0) != 0)[0])
        
        # Calculate minimum-norm solution at active dipoles and patches
        x_hat = np.zeros((n_dipoles, n_time))

        ## Use different y without unnecessary components here:
        x_hat[omegas, :] = np.linalg.pinv(self.leadfield[:, omegas]) @ y
        return x_hat

    def calc_smp_solution(self, y, include_singletons=True, residual_thresh=0.01):
        """ Calculates the Orthogonal Matching Pursuit (OMP) inverse solution.
        
        Parameters
        ----------
        y : numpy.ndarray
            The data matrix (channels,).
        include_singletons : bool
            If True -> Include not only smooth patches but also single dipoles.

        Return
        ------
        x_hat : numpy.ndarray
            The inverse solution (dipoles,)
        """
        n_chans = len(y)
        _, n_dipoles = self.leadfield.shape
        
        y -= y.mean()
        x_hat = np.zeros(n_dipoles)
        omega = np.array([])
        r = deepcopy(y)
        y_hat = self.leadfield@x_hat
        y_hat -= y_hat.mean(axis=0)
        residuals = np.array([np.linalg.norm(y - y_hat), ])
        unexplained_variance = np.array([calc_residual_variance(y_hat, y),])
        # source_norms = np.array([0,])
        x_hats = [deepcopy(x_hat), ]

        for _ in range(n_chans):
            b_smooth = self.leadfield_smooth_normed.T @ r
            b_sparse = self.leadfield_normed.T @ r

            if include_singletons & (abs(b_sparse).max() > abs(b_smooth).max()):  # if sparse is better
                b_sparse_thresh = thresholding(b_sparse, 1)
                new_patch = np.where(b_sparse_thresh != 0)[0]
                
            else: # else if patch is better
                b_smooth_thresh = thresholding(b_smooth, 1)
                new_patch = np.where(self.laplace_operator[b_smooth_thresh!=0][0]!=0)[0]


            omega = np.append(omega, new_patch)
            omega = omega.astype(int)
            x_hat[omega] = np.linalg.pinv(self.leadfield[:, omega]) @ y
            y_hat = self.leadfield@x_hat
            y_hat -= y_hat.mean(axis=0)
            r = y - y_hat

            residuals = np.append(residuals, np.linalg.norm(r))
            unexplained_variance = np.append(unexplained_variance, calc_residual_variance(y_hat, y))
            # source_norms = np.append(source_norms, np.sum(x_hat**2))
            x_hats.append(deepcopy(x_hat))
            if residuals[-1] > residuals[-2]:
                break



        
        # Remove where residual is below threshold:
        # keep_idc = np.where(unexplained_variance>=residual_thresh*100)[0]
        # if len(keep_idc) == 1:
        #     keep_idc = np.append(keep_idc, 1)
        
        # print(keep_idc)
        # unexplained_variance = unexplained_variance[keep_idc]
        # x_hats = [x_hats[idx] for idx in keep_idc]
        
        x_hat = best_index_residual(unexplained_variance, x_hats, plot=False)
        
        # x_hat = x_hats[corner_idx]
        return x_hat