import numpy as np
import mne
from copy import deepcopy
from scipy.sparse.csgraph import laplacian
from ..util import find_corner
from .base import BaseSolver, InverseOperator

class SolverMUSIC(BaseSolver):
    ''' Class for the Multiple Signal Classification (MUSIC) inverse solution
        [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Baillet, S., Mosher, J. C., & Leahy, R. M. (2001). Electromagnetic brain
    mapping. IEEE Signal processing magazine, 18(6), 14-30.
    
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
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, n="auto", stop_crit=0.95) -> mne.SourceEstimate:
        ''' Apply MUSIC inverse solution.
        
        Parameters
        ----------
        evoked : mne.Evoked
            The evoked data object.
        stop_crit : float
            Controls the percentage of top active dipoles that are selected
            (i.e., sparsity).

        Return
        ------
        stc : mne.SourceEstimate
            The inverse solution source estimate object.
        '''
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
        
        # Get optimal eigenvectors
        U, D, _ = np.linalg.svd(C, full_matrices=False)
        if n == "auto":
            # L-curve method
            # D = D[:int(n_chans/2)]
            # iters = np.arange(len(D))
            # n_comp = find_corner(deepcopy(iters), deepcopy(D))
            
            # eigenvalue magnitude-based
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(((D**2)*len((D**2)) / (D**2).sum()))
            # n_comp = np.where(((D**2)*len((D**2)) / (D**2).sum()) < np.exp(-16))[0][0]
            D_ = D/D.max()
            n_comp = np.where( abs(np.diff(D_)) < 0.01 )[0][0]+1
        else:
            n_comp = deepcopy(n)
        Us = U[:, :n_comp]
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
    (RAP-MUSIC) inverse solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Mosher, J. C., & Leahy, R. M. (1999). Source localization using
    recursively applied and projected (RAP) MUSIC. IEEE Transactions on signal
    processing, 47(2), 332-340.
    '''
    def __init__(self, name="RAP-MUSIC", **kwargs):
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
        leadfield = self.leadfield        
        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked, n="auto", k="auto", stop_crit=0.95) -> mne.SourceEstimate:
        ''' Apply RAP-MUSIC inverse solution.
        
        Parameters
        ----------
        evoked : mne.Evoked
            The evoked data object.
        n : ["auto", int]
            Number of eigenvectors to use.
        k : int
            Number of recursions.
        stop_crit : float
            Controls the percentage of top active dipoles that are selected
            (i.e., sparsity).

        Return
        ------
        stc : mne.SourceEstimate
            The inverse solution source estimate object.
        '''
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
        
        if k == "auto":
            k = n_chans

        # Data Covariance
        C = y@y.T
        I = np.identity(n_chans)
        Q = np.identity(n_chans)

        U, D, _= np.linalg.svd(C, full_matrices=True)
        if n == "auto":
            # L-curve method
            # D = D[:int(n_chans/2)]
            # iters = np.arange(len(D))
            # n_comp = find_corner(deepcopy(iters), deepcopy(D))
            
            # eigenvalue magnitude-based
            # n_comp = np.where(((D**2)*len((D**2)) / (D**2).sum()) < np.exp(-16))[0][0]

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(iters, D, '*k')
            # plt.plot(iters[n_comp], D[n_comp], 'or')
            # plt.plot(iters[n_comp], D[n_comp], 'og')
            D_ = D/D.max()
            n_comp = np.where( abs(np.diff(D_)) < 0.01 )[0][0]+1

        else:
            n_comp = deepcopy(n)
        Us = U[:, :n_comp]
        dipole_idc = []
        n_time = y.shape[1]
        for i in range(k):
            # print(i)
            
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

            U, D, _= np.linalg.svd(C, full_matrices=False)
            Us = U[:, :n_comp]

        dipole_idc = np.array(dipole_idc)
        x_hat = np.zeros((n_dipoles, n_time))
        x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        return x_hat

class SolverTRAPMUSIC(BaseSolver):
    ''' Class for the Truncated Recursively Applied Multiple Signal
        Classification (TRAP-MUSIC) inverse solution [1].
    
    Attributes
    ----------
    
    References
    ----------
    [1] Mäkelä, N., Stenroos, M., Sarvas, J., & Ilmoniemi, R. J. (2018).
    Truncated rap-music (trap-music) for MEG and EEG source localization.
    NeuroImage, 167, 73-83.
    '''
    def __init__(self, name="TRAP-MUSIC", **kwargs):
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
        leadfield = self.leadfield
        self.inverse_operators = []

        return self

    def apply_inverse_operator(self, evoked, n="auto", k="auto", stop_crit=0.95) -> mne.SourceEstimate:
        ''' Apply TRAP-MUSIC inverse solution.
        
        Parameters
        ----------
        evoked : mne.Evoked
            The evoked data object.
        n : ["auto", int]
            Number of eigenvectors to use.
        k : int
            Number of recursions.
        stop_crit : float
            Controls the percentage of top active dipoles that are selected
            (i.e., sparsity).

        Return
        ------
        stc : mne.SourceEstimate
            The inverse solution source estimate object.
        '''
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
        
        if k == "auto":
            k = n_chans
        
        # Data Covariance
        C = y@y.T
        I = np.identity(n_chans)
        Q = np.identity(n_chans)
        U, D, _= np.linalg.svd(C, full_matrices=True)
        if n == "auto":
            # L-curve method
            # D = D[:int(n_chans/2)]
            # iters = np.arange(len(D))
            # n_comp = find_corner(deepcopy(iters), deepcopy(D))
            
            # eigenvalue magnitude-based
            # n_comp = np.where(((D**2)*len((D**2)) / (D**2).sum()) < np.exp(-16))[0][0]

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(iters, D, '*k')
            # plt.plot(iters[n_comp], D[n_comp], 'or')
            # plt.plot(iters[n_comp], D[n_comp], 'og')
            D_ = D/D.max()
            n_comp = np.where( abs(np.diff(D_)) < 0.01 )[0][0]+1

        else:
            n_comp = deepcopy(n)
        Us = U[:, :n_comp]
        dipole_idc = []
        n_time = y.shape[1]
        for i in range(k):
            # print(i)
            
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
            U, D, _= np.linalg.svd(C, full_matrices=False)
            Us = U[:, :n_comp-i]
        dipole_idc = np.array(dipole_idc)
        x_hat = np.zeros((n_dipoles, n_time))
        x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        return x_hat

class SolverJAZZMUSIC(BaseSolver):
    ''' Class for the Smooth RAP Multiple Signal Classification (JAZZ-MUSIC)
        inverse solution.
    
    Attributes
    ----------
    
    References
    ---------
    This method is of my own making (Lukas Hecker, 2022) and unpublished.

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
        n_orders : int
            Controls the maximum smoothness to pursue.

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

    def apply_inverse_operator(self, evoked, n="auto", k="auto", stop_crit=0.95, truncate=True) -> mne.SourceEstimate:
        ''' Apply JAZZ-MUSIC inverse solution.
        
        Parameters
        ----------
        evoked : mne.Evoked
            The evoked data object.
        n : ["auto", int]
            Number of eigenvectors to use.
        k : int
            Number of recursions.
        stop_crit : float
            Controls the percentage of top active dipoles that are selected
            (i.e., sparsity).
        truncate : bool
            If True: Truncate SVD's eigenvectors (like TRAP-MUSIC), otherwise
            don't (like RAP-MUSIC).

        Return
        ------
        stc : mne.SourceEstimate
            The inverse solution source estimate object.
        '''
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
        if k == "auto":
            k = n_chans
        # Data Covariance
        y -= y.mean(axis=0)

        C = y@y.T


        I = np.identity(n_chans)
        Q = np.identity(n_chans)
        U, D, _= np.linalg.svd(C, full_matrices=True)
        # U -= U.mean(axis=0)

        if n == "auto":
            # L-curve method
            # D = D[:int(n_chans)]
            # iters = np.arange(len(D))
            # n_comp_L = find_corner(deepcopy(iters), deepcopy(D)) + 1
            
            # eigenvalue magnitude-based
            # n_comp_spm = np.where(((D**2)*len((D**2)) / (D**2).sum()) < np.exp(-16))[0][0]

            
            
            # Based on eigenvalue drop-off
            D_ = D/D.max()
            n_comp = np.where( abs(np.diff(D_)) < 0.001 )[0][0] + 2
            # plt.plot(iters[n_comp], D_[n_comp], 'ob')
            # print("Spatial Components: ", n_comp)


            # import matplotlib.pyplot as plt
            # iters = np.arange(len(D))
            # D_ = D/D.max()
            # plt.figure()
            # plt.plot(iters, D_, '*k')
            # plt.plot(iters[n_comp], D_[n_comp], 'og', label="Eig drop-off")
            # # plt.plot(iters[n_comp_spm], D_[n_comp_spm], 'or', label="SPM Method")
            # plt.plot(iters[n_comp_L], D_[n_comp_L], 'ob', label="L Curve Method")
            
        else:
            n_comp = deepcopy(n)
        Us = U[:, :n_comp]

        dipole_idc = []
        n_time = y.shape[1]
        for i in range(k):
            # print(Us.shape)
            Ps = Us@Us.T

            mu = np.zeros((n_orders, n_dipoles))
            # for nn in range(n_orders):
            #     for p in range(n_dipoles):
            #         l = leadfields[nn][:, p][:, np.newaxis]
            #         norm_1 = np.linalg.norm(Ps @ Q @ l)
            #         norm_2 = np.linalg.norm(Q @ l) # Q @ l
            #         mu[nn, p] = norm_1 / norm_2
            for nn in range(n_orders):
                norm_1 = np.linalg.norm(Ps @ Q @ leadfields[nn], axis=0)
                norm_2 = np.linalg.norm(Q @ leadfields[nn], axis=0) 
                mu[nn, :] = norm_1 / norm_2
        
            # Find the dipole/ patch with highest correlation with the residual
            best_order, best_dipole = np.unravel_index(np.argmax(mu), mu.shape)
            # print("Best order: ", best_order)
            # Add dipole index or patch indices to the list of active dipoles
            dipole_idx = self.neighbors[best_order][best_dipole]
            dipole_idc.extend( dipole_idx )

            if np.max(mu) < stop_crit:
                # print("stopping at ", np.max(mu))
                break

            if i == 0:
                # B = leadfield[:, dipole_idx]
                B = leadfields[best_order][:, best_dipole][:, np.newaxis]
            else:
                # B = np.hstack([B, leadfield[:, dipole_idx]])
                B = np.hstack([B, leadfields[best_order][:, best_dipole][:, np.newaxis]])

            # B = B / np.linalg.norm(B, axis=0)
            Q = I - B @ np.linalg.pinv(B)
            # Q -= Q.mean(axis=0)
            C = Q @ Us

            U, D, _= np.linalg.svd(C, full_matrices=False)
            # U -= U.mean(axis=0)
            
        
            # 
            if truncate:
                Us = U[:, :n_comp-i]
            else:
                Us = U[:, :n_comp]
            
        dipole_idc = np.array(dipole_idc)

        # Simple minimum norm inversion using found dipoles
        x_hat = np.zeros((n_dipoles, n_time))
        
        # Time-course estimation
        
        # Simple MNE-based 
        # x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y

        # WMNE-based
        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y

        return x_hat

    def make_jazz(self, n_orders):
        ''' Create the dictionary of increasingly smooth sources.
        
        Parameters
        ----------
        n_orders : int
            Number of neighborhood orders to include to the dictionary. The
            higher, the smoother the sources can be.
        
        '''
        n_dipoles = self.leadfield.shape[1]
        

        self.leadfields = [deepcopy(self.leadfield), ]
        self.neighbors = [[np.array([i]) for i in range(n_dipoles)], ]

        if n_orders==0:
            return

        new_leadfield = deepcopy(self.leadfield)
        self.adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        self.gradient = abs(laplacian(self.adjacency))
        new_adjacency = deepcopy(self.adjacency)

        
        for _ in range(n_orders):
            new_leadfield = new_leadfield @ self.gradient
            new_leadfield -= new_leadfield.mean(axis=0)
            new_leadfield /= np.linalg.norm(new_leadfield, axis=0)

            new_adjacency = new_adjacency @ self.adjacency
            neighbors = [np.where(ad!=0)[0] for ad in self.adjacency.toarray()]
            
            self.leadfields.append( deepcopy(new_leadfield) )
            self.neighbors.append( neighbors )