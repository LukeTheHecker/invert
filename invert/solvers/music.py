import numpy as np
import mne
from copy import deepcopy
from scipy.sparse.csgraph import laplacian
from ..util import find_corner
from .base import BaseSolver, InverseOperator

class SolverMUSIC(BaseSolver):
    ''' Class for the Multiple Signal Classification (MUSIC) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="MUSIC", **kwargs):
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
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, n=6, stop_crit=0.95) -> mne.SourceEstimate:
        source_mat = self.apply_music(evoked.data, n, stop_crit)
        stc = self.source_to_object(source_mat, evoked)
        return stc

    def apply_music(self, y, n, stop_crit):
        ''' Apply the MUSIC inverse solution to the EEG data.
        
        Parameters
        ----------
        y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int
            Number of eigenvectors to use.
        stop_crit : float
            Criterion at which to select candidate dipoles. The lower, the more
            dipoles will be incorporated.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        n_dipoles = self.leadfield.shape[1]
        n_time = y.shape[1]
        
        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        
        # Data Covariance
        C = y@y.T
        U, D, _  = np.linalg.svd(C, full_matrices=False)
        Us = U[:, :n]
        Ps = Us@Us.T

        mu = np.zeros(n_dipoles)
        for p in range(n_dipoles):
            l = leadfield[:, p][:, np.newaxis]
            norm_1 = np.linalg.norm(Ps @ l)
            norm_2 = np.linalg.norm(l)
            mu[p] = norm_1 / norm_2
        mu[mu<stop_crit] = 0

        dipole_idc = np.where(mu!=0)[0]
        x_hat = np.zeros((n_dipoles, n_time))
        x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        return x_hat


class SolverRAPMUSIC(BaseSolver):
    ''' Class for the Recursively Applied Multiple Signal Classification
    (RAP-MUSIC) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="RAP-MUSIC", **kwargs):
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
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, n=10, k=5, stop_crit=0.95) -> mne.SourceEstimate:
        source_mat = self.apply_rapmusic(evoked.data, n, k, stop_crit)
        stc = self.source_to_object(source_mat, evoked)
        return stc

    def apply_rapmusic(self, y, n, k, stop_crit):
        ''' Apply the RAP-MUSIC inverse solution to the EEG data.
        
        Parameters
        ----------
        y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int
            Number of eigenvectors to use.
        k : int
            Number of recursions.
        stop_crit : float
            Criterion to stop recursions. The lower, the more
            dipoles will be incorporated.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        n_chans, n_dipoles = self.leadfield.shape
        n_time = y.shape[1]
        
        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        
        # Data Covariance
        C = y@y.T
        I = np.identity(n_chans)
        Q = np.identity(n_chans)
        dipole_idc = []
        n_time = y.shape[1]
        for i in range(k):
            # print(i)
            U, D, _= np.linalg.svd(C, full_matrices=False)
            Us = U[:, :n]
            Ps = Us@Us.T

            mu = np.zeros(n_dipoles)
            for p in range(n_dipoles):
                l = leadfield[:, p][:, np.newaxis]
                norm_1 = np.linalg.norm(Ps @ Q @ l)
                norm_2 = np.linalg.norm(Q @ l)
                
                mu[p] = norm_1 / norm_2
            dipole_idx = np.argmax(mu)
            dipole_idc.append( dipole_idx )

            if np.max(mu) < stop_crit:
                # print("breaking")
                break

            if i == 0:
                B = leadfield[:, dipole_idx][:, np.newaxis]
            else:
                B = np.hstack([B, leadfield[:, dipole_idx][:, np.newaxis]])
            
            Q = I - B @ np.linalg.pinv(B)
            C = Q @ Us

        dipole_idc = np.array(dipole_idc)
        x_hat = np.zeros((n_dipoles, n_time))
        x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        return x_hat

class SolverTRAPMUSIC(BaseSolver):
    ''' Class for the Truncated Recursively Applied Multiple Signal Classification
    (TRAP-MUSIC) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="TRAP-MUSIC", **kwargs):
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
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, n=10, k=5, stop_crit=0.95) -> mne.SourceEstimate:
        source_mat = self.apply_trapmusic(evoked.data, n, k, stop_crit)
        stc = self.source_to_object(source_mat, evoked)
        return stc

    def apply_trapmusic(self, y, n, k, stop_crit):
        ''' Apply the TRAP-MUSIC inverse solution to the EEG data.
        
        Parameters
        ----------
        y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int
            Number of eigenvectors to use.
        k : int
            Number of recursions.
        stop_crit : float
            Criterion to stop recursions. The lower, the more
            dipoles will be incorporated.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        
        n_chans, n_dipoles = self.leadfield.shape
        n_time = y.shape[1]
        
        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        
        # Data Covariance
        C = y@y.T
        I = np.identity(n_chans)
        Q = np.identity(n_chans)

        dipole_idc = []
        n_time = y.shape[1]
        for i in range(k):
            # print(i)
            U, D, _= np.linalg.svd(C, full_matrices=False)
            Us = U[:, :n-i]
            Ps = Us@Us.T

            mu = np.zeros(n_dipoles)
            for p in range(n_dipoles):
                l = leadfield[:, p][:, np.newaxis]
                norm_1 = np.linalg.norm(Ps @ Q @ l)
                norm_2 = np.linalg.norm(Q @ l)
                mu[p] = norm_1 / norm_2
            dipole_idx = np.argmax(mu)
            dipole_idc.append( dipole_idx )

            if np.max(mu) < stop_crit:
                # print("breaking")
                break

            if i == 0:
                B = leadfield[:, dipole_idx][:, np.newaxis]
            else:
                B = np.hstack([B, leadfield[:, dipole_idx][:, np.newaxis]])
            
            Q = I - B @ np.linalg.pinv(B)
            C = Q @ Us

        dipole_idc = np.array(dipole_idc)
        x_hat = np.zeros((n_dipoles, n_time))
        x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        return x_hat

class SolverJAZZMUSIC(BaseSolver):
    ''' Class for the Smooth RAP Multiple Signal Classification (JAZZ-MUSIC) inverse
    solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="JAZZ-MUSIC", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, alpha="auto", n_orders=3, verbose=0, **kwargs):
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

        self.make_jazz(n_orders)
        

        n_chans, _ = leadfield.shape
        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, n="auto", k=5, stop_crit=0.95, truncate=True) -> mne.SourceEstimate:
        source_mat = self.apply_jazzmusic(evoked.data, n, k, stop_crit, truncate)
        stc = self.source_to_object(source_mat, evoked)
        return stc

    def apply_jazzmusic(self, y, n, k, stop_crit, truncate):
        ''' Apply the RAP-MUSIC inverse solution to the EEG data.
        
        Parameters
        ----------
        y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int/ str
            Number of eigenvectors to use or "auto" for l-curve method.
        k : int
            Number of recursions.
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.
        truncate : bool
            If True: Truncate SVD's eigenvectors (like TRAP-MUSIC), otherwise
            don't (like RAP-MUSIC).
        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        n_chans, n_dipoles = self.leadfield.shape
        n_time = y.shape[1]

        leadfield = self.leadfield
        leadfield -= leadfield.mean(axis=0)
        
        leadfields = self.leadfields
        n_orders = len(self.leadfields)
        
        # Data Covariance
        C = y@y.T
        I = np.identity(n_chans)
        Q = np.identity(n_chans)
        U, D, _= np.linalg.svd(C, full_matrices=True)
        if n == "auto":
            iters = np.arange(len(D))
            n_comp = find_corner(deepcopy(iters), deepcopy(D))
        else:
            n_comp = deepcopy(n)
        Us = U[:, :n_comp]

        dipole_idc = []
        n_time = y.shape[1]
        for i in range(k):
            Ps = Us@Us.T

            mu = np.zeros((n_orders, n_dipoles))
            for nn in range(n_orders):
                for p in range(n_dipoles):
                    l = leadfields[nn][:, p][:, np.newaxis]
                    norm_1 = np.linalg.norm(Ps @ Q @ l)
                    norm_2 = np.linalg.norm(Q @ l)
                    mu[nn, p] = norm_1 / norm_2
            
            # Find the dipole/ patch with highest correlation with the residual
            best_order, best_dipole = np.unravel_index(np.argmax(mu), mu.shape)
            
            # Add dipole index or patch indices to the list of active dipoles
            dipole_idx = self.neighbors[best_order][best_dipole]
            dipole_idc.extend( dipole_idx )

            if np.max(mu) < stop_crit:
                break

            if i == 0:
                B = leadfield[:, dipole_idx]
                # B = leadfields[best_order][:, best_dipole][:, np.newaxis]
            else:
                B = np.hstack([B, leadfield[:, dipole_idx]])
                # B = np.hstack([B, leadfields[best_order][:, best_dipole][:, np.newaxis]])

            # B = B / np.linalg.norm(B, axis=0)
            Q = I - B @ np.linalg.pinv(B)
            C = Q @ Us

            U, D, _= np.linalg.svd(C, full_matrices=False)

            # 
            if truncate:
                Us = U[:, :n_comp-i]
            else:
                Us = U[:, :n_comp]

        dipole_idc = np.array(dipole_idc)
        x_hat = np.zeros((n_dipoles, n_time))
        x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        return x_hat

    def make_jazz(self, n_orders):
        n_dipoles = self.leadfield.shape[1]
        self.adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        self.gradient = abs(laplacian(self.adjacency))
        
        # Normalized Gradient
        # self.gradient = abs(laplacian(self.adjacency)).toarray().astype(np.float64)
        # self.gradient = np.stack([g / np.mean(g[g!=0]) for g in self.gradient], axis=0)


        new_leadfield = deepcopy(self.leadfield)
        new_adjacency = deepcopy(self.adjacency)

        self.leadfields = [deepcopy(self.leadfield), ]
        self.neighbors = [[np.array([i]) for i in range(n_dipoles)], ]

        for i in range(n_orders):
            new_leadfield = new_leadfield @ self.gradient
            new_leadfield -= new_leadfield.mean(axis=0)
            new_adjacency = new_adjacency @ self.adjacency
            neighbors = [np.where(ad!=0)[0] for ad in self.adjacency.toarray()]
            
            self.leadfields.append( new_leadfield )
            self.neighbors.append( neighbors )

