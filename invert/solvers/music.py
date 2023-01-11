import numpy as np
import mne
from copy import deepcopy
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cdist
from ..util import find_corner, pos_from_forward
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

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", n="auto", stop_crit=0.95, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        n : int/ str
            Number of eigenvectors to use or "auto" for l-curve method.
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.

        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)
        inverse_operator = self.make_music(data, n, stop_crit)
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_music(self, y, n, stop_crit):
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
        n_chans, n_dipoles = self.leadfield.shape
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
        # x_hat = np.zeros((n_dipoles, n_time))
        # x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
        # return x_hat

        # WMNE-based
        # x_hat = np.zeros((n_dipoles, n_time))
        inverse_operator = np.zeros((n_dipoles, n_chans))

        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        
        inverse_operator[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T

        return inverse_operator


class SolverFLEXMUSIC(BaseSolver):
    ''' Class for the RAP Multiple Signal Classification with flexible extent
        estimation (FLEX-MUSIC).
    
    Attributes
    ----------
    n_orders : int
        Controls the maximum smoothness to pursue.
    truncate : bool
            If True: Truncate SVD's eigenvectors (like TRAP-MUSIC), otherwise
            don't (like RAP-MUSIC).

    References
    ---------
    This method is of my own making (Lukas Hecker, 2022) and soon to be
    published.

    '''
    def __init__(self, name="FLEX-MUSIC", n_orders=3, truncate=False, **kwargs):
        self.name = name
        self.n_orders = n_orders
        self.truncate = truncate
        self.is_prepared = False
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", n="drop", k="auto", stop_crit=0.975, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        
        n : int/ str
            Number of eigenvalues to use.
                int: The number of eigenvalues to use.
                "L": L-curve method for automated selection.
                "drop": Selection based on relative change of eigenvalues.
                "auto": Combine L and drop method
                "mean": Selects the eigenvalues that are larger than the mean of all eigs.
        k : int
            Number of recursions.
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.
        

        Return
        ------
        self : object returns itself for convenience
        '''
        from time import time
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        
        data = self.unpack_data_obj(mne_obj)
        if not self.is_prepared:
            self.prepare_flex()
        
        inverse_operator = self.make_flex(data, n, k, stop_crit, self.truncate)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_flex(self, y, n, k, stop_crit, truncate):
        ''' Create the FLEX-MUSIC inverse solution to the EEG data.
        
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
        # Assert common average reference
        y -= y.mean(axis=0)
        # Compute Data Covariance
        C = y@y.T
        
        I = np.identity(n_chans)
        Q = np.identity(n_chans)
        U, D, _= np.linalg.svd(C, full_matrices=False)
        
        if type(n) == str:
            if n == "L":
                # Based L-Curve Criterion
                n_comp = self.get_comps_L(D)

            elif n == "drop":
                # Based on eigenvalue drop-off
                n_comp = self.get_comps_drop(D)
            elif n == "mean":
                n_comp = np.where(D<D.mean())[0][0]
                
            elif n == "auto":
                n_comp_L = self.get_comps_L(D)
                n_comp_drop = self.get_comps_drop(D)
                
                # Combine the two:
                n_comp = np.ceil((n_comp_drop + n_comp_L)/2).astype(int)
        else:
            n_comp = deepcopy(n)

        Us = U[:, :n_comp]

        dipole_idc = []
        source_covariance = np.zeros(n_dipoles)
        
        for i in range(k):
            # print(Us.shape)
            Ps = Us@Us.T

            mu = np.zeros((n_orders, n_dipoles))
            for nn in range(n_orders):
                norm_1 = np.linalg.norm(Ps @ Q @ leadfields[nn], axis=0)
                norm_2 = np.linalg.norm(Q @ leadfields[nn], axis=0) 
                mu[nn, :] = norm_1 / norm_2

            
                
            # Find the dipole/ patch with highest correlation with the residual
            best_order, best_dipole = np.unravel_index(np.argmax(mu), mu.shape)
            
            if i>0 and np.max(mu) < stop_crit:
                # print("stopping at ", np.max(mu))
                break

            # source_covariance += np.squeeze(self.gradients[best_order][best_dipole] * (1/np.sqrt(i+1)))
            source_covariance += np.squeeze(self.gradients[best_order][best_dipole])

            if i == 0:
                B = leadfields[best_order][:, best_dipole][:, np.newaxis]
            else:
                B = np.hstack([B, leadfields[best_order][:, best_dipole][:, np.newaxis]])

            # B = B / np.linalg.norm(B, axis=0)
            Q = I - B @ np.linalg.pinv(B)
            # Q -= Q.mean(axis=0)
            C = Q @ Us

            U, D, _= np.linalg.svd(C, full_matrices=False)
            # U -= U.mean(axis=0)
            
        
            # Truncate eigenvectors
            if truncate:
                Us = U[:, :n_comp-i]
            else:
                Us = U[:, :n_comp]
            
        dipole_idc = np.array(dipole_idc).astype(int)
        n_time = y.shape[1]
        # # Simple minimum norm inversion using found dipoles
        # x_hat = np.zeros((n_dipoles, n_time))
        # x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y

        # # WMNE-based - use the selected dipole indices and calc WMNE solution
        # # x_hat = np.zeros((n_dipoles, n_time))
        # inverse_operator = np.zeros((n_dipoles, n_chans))
        # L = self.leadfield[:, dipole_idc]
        # W = np.diag(np.linalg.norm(L, axis=0))
        # # x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y
        # inverse_operator[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T

        # # Prior-Cov based: Use the selected smooth patches as source covariance priors
        # print("non-zero dipoles in source cov: ", (source_covariance!=0).sum())
        # source_covariance = np.diag(source_covariance)
        # L = self.leadfield
        # W = np.diag(np.linalg.norm(L, axis=0)) 
        # # print(source_covariance.shape, L.shape, W.shape)
        # inverse_operator = source_covariance @ np.linalg.inv(source_covariance @ L.T @ L + W.T @ W) @ source_covariance @ L.T
        
        # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
        source_covariance = csr_matrix(np.diag(source_covariance))
        L_s = self.leadfield @ source_covariance
        L = self.leadfield
        W = np.diag(np.linalg.norm(L, axis=0)) 
        # print(source_covariance.shape, L.shape, W.shape)
        inverse_operator = source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T

        # # Prior-Cov dSPM-based:
        # source_covariance = np.diag(source_covariance)
        # leadfield_source_cov = source_covariance @ self.leadfield.T
        # LLS = self.leadfield @ leadfield_source_cov
        # K = leadfield_source_cov @ np.linalg.inv(LLS + 0.0001*np.identity(n_chans))
        # # W_dSPM = np.diag( 1 / np.sqrt( np.diagonal(K @ np.identity(n_chans) @ K.T) ) )
        # W_dSPM =  1 / np.sqrt( np.diagonal(K @ np.identity(n_chans) @ K.T) )
        # inverse_operator = (K.T * W_dSPM).T

        # # sLORETA based
        # source_covariance = np.diag(source_covariance)
        # L_s = self.leadfield @ source_covariance
        # LLT = L_s @ L_s.T
        # K_MNE = leadfield.T @ np.linalg.pinv(LLT)
        # W_diag = np.sqrt(np.diag(K_MNE @ L_s))
        # inverse_operator = (K_MNE.T / W_diag).T

        # # GAMMA based:
        # Gamma = np.diag(source_covariance)
        # Sigma_y = leadfield @ Gamma @ leadfield.T
        # Sigma_y_inv = np.linalg.inv(Sigma_y)
        # inverse_operator = Gamma @ leadfield.T @ Sigma_y_inv

        return inverse_operator

    def prepare_flex(self):
        ''' Create the dictionary of increasingly smooth sources unless self.n_orders==0.
        
        Parameters
        ----------

        
        '''
        n_dipoles = self.leadfield.shape[1]
        
        self.leadfields = [deepcopy(self.leadfield), ]
        # self.neighbors = [[np.array([i]) for i in range(n_dipoles)], ]
        self.gradients = [np.identity(n_dipoles),]

        if self.n_orders==0:
            return

        new_leadfield = deepcopy(self.leadfield)
        self.adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        gradient = abs(laplacian(deepcopy(self.adjacency)))
        gradient = csr_matrix(gradient.toarray() / gradient.toarray().max(axis=0))
        # Convert to sparse matrix for speedup
        gradient = csr_matrix(gradient)
        
        for _ in range(self.n_orders):
            # new_leadfield = new_leadfield @ gradient
            new_leadfield = self.leadfield @ gradient
            # new_leadfield -= new_leadfield.mean(axis=0)
            # new_leadfield /= np.linalg.norm(new_leadfield, axis=0)
            
            # neighbors = [np.where(ad!=0)[0] for ad in gradient.toarray()]
            
            self.leadfields.append( deepcopy(new_leadfield) )
            # self.neighbors.append( neighbors )
            self.gradients.append( gradient.toarray() )

            gradient = gradient @ deepcopy(self.adjacency)
            gradient = csr_matrix(gradient.toarray() / gradient.toarray().max(axis=0))
        
        self.is_prepared = True

    @staticmethod
    def get_comps_L(D):
        # L-curve method
        iters = np.arange(len(D))
        n_comp_L = find_corner(deepcopy(iters), deepcopy(D))
        return n_comp_L
    @staticmethod
    def get_comps_drop(D):
        D_ = D/D.max()
        n_comp_drop = np.where( abs(np.diff(D_)) < 0.001 )[0]

        if len(n_comp_drop) > 0:
            n_comp_drop = n_comp_drop[0] + 1
        else:
            n_comp_drop = 1
        return n_comp_drop         

class SolverFLEXMUSIC_2(BaseSolver):
    ''' Class for the RAP Multiple Signal Classification with flexible extent
        estimation (FLEX-MUSIC).
    
    Attributes
    ----------
    n_orders : int
        Controls the maximum smoothness to pursue.
    truncate : bool
            If True: Truncate SVD's eigenvectors (like TRAP-MUSIC), otherwise
            don't (like RAP-MUSIC).

    References
    ---------
    This method is of my own making (Lukas Hecker, 2022) and soon to be
    published.

    '''
    def __init__(self, name="FLEX-MUSIC 2", truncate=False, **kwargs):
        self.name = name
        self.truncate = truncate
        self.is_prepared = False
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", n="drop", k="auto", stop_crit=0.975, verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        alpha : float
            The regularization parameter.
        
        n : int/ str
            Number of eigenvalues to use.
                int: The number of eigenvalues to use.
                "L": L-curve method for automated selection.
                "drop": Selection based on relative change of eigenvalues.
                "auto": Combine L and drop method
                "mean": Selects the eigenvalues that are larger than the mean of all eigs.
        k : int
            Number of recursions.
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.
        

        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        
        data = self.unpack_data_obj(mne_obj)
        if not self.is_prepared:
            self.prepare_flex()
        
        inverse_operator = self.make_flex(data, n, k, stop_crit, self.truncate)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_flex(self, y, n, k, stop_crit, truncate):
        ''' Create the FLEX-MUSIC inverse solution to the EEG data.
        
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

        if k == "auto":
            k = n_chans
        # Assert common average reference
        y -= y.mean(axis=0)
        # Compute Data Covariance
        C = y@y.T
        
        I = np.identity(n_chans)
        Q = np.identity(n_chans)
        U, D, _= np.linalg.svd(C, full_matrices=False)
        
        if type(n) == str:
            if n == "L":
                # Based L-Curve Criterion
                n_comp = self.get_comps_L(D)

            elif n == "drop":
                # Based on eigenvalue drop-off
                n_comp = self.get_comps_drop(D)
            elif n == "mean":
                n_comp = np.where(D<D.mean())[0][0]
                
            elif n == "auto":
                n_comp_L = self.get_comps_L(D)
                n_comp_drop = self.get_comps_drop(D)
                
                # Combine the two:
                n_comp = np.ceil((n_comp_drop + n_comp_L)/2).astype(int)
        else:
            n_comp = deepcopy(n)
        self.n_comp = n_comp
        Us = U[:, :n_comp]

        source_covariance = np.zeros(n_dipoles)
        
        for i in range(k):
            # print(Us.shape)
            Ps = Us@Us.T
            selected_idc = []

            norm_1 = np.linalg.norm(Ps @ Q @ leadfield, axis=0)
            norm_2 = np.linalg.norm(Q @ leadfield, axis=0) 
            mu = norm_1 / norm_2
            selected_idc.append( np.argmax(mu) )
            current_max = np.max(mu)
            current_leadfield = leadfield[:, selected_idc[0]]
            component_strength = [1,]
            # print("initial idx: ", selected_idc[0])
            # print("initial max: ", current_max)

            while True:
                # Find all neighboring dipoles of current candidates
                neighbors = np.unique(np.concatenate([self.members_ex[idx] for idx in selected_idc]))
                # Filter out candidates from neighbors
                for idx, n in reversed(list(enumerate(neighbors))):
                    if n in selected_idc:
                        # print("del")
                        neighbors = np.delete(neighbors, idx)

                dist = self.distances[neighbors, selected_idc[0]]

                # construct new candidate leadfields:
                b = np.stack([ leadfield[:, n]/d + current_leadfield for d, n in zip(dist, neighbors)], axis=1)
                # b /= np.linalg.norm(b, axis=0)
                # norm_1 = np.linalg.norm(Ps @ Q @ b, axis=0)
                # norm_2 = np.linalg.norm(Q @ b, axis=0) 
                norm_1 = np.linalg.norm(abs(Ps) @ abs(Q) @ abs(b), axis=0)
                norm_2 = np.linalg.norm(abs(Q) @ abs(b), axis=0) 
                
                mu_b = norm_1 / norm_2
                # print(mu_b)
                max_mu_b = np.max(mu_b)
                
                if (max_mu_b - current_max) < 0.00:
                # if max_mu_b / current_max < 1.0001:
                    # print("stop, change is ", max_mu_b - current_max)
                    break
                else:
                    new_idx = neighbors[np.argmax(mu_b)]
                    b_best = b[:, np.argmax(mu_b)]
                    # print("added index ", new_idx)
                    # print("current max: ", max_mu_b)
                    current_max = max_mu_b
                    current_leadfield = b_best #/ np.linalg.norm(b_best)
                    component_strength.append( 1/dist[np.argmax(mu_b)])
                    selected_idc.append( new_idx )
                            
            # Find the dipole/ patch with highest correlation with the residual
            
            if i>0 and current_max < stop_crit:
                # print("stopping at ", current_max)
                break
            selected_idc = np.array(selected_idc)
            current_cov = np.zeros(n_dipoles)
            current_cov[selected_idc] = np.array(component_strength)
            source_covariance += current_cov
            # source_covariance += np.squeeze(self.identity[selected_idc].sum(axis=0))
            current_leadfield /= np.linalg.norm(current_leadfield)
            if i == 0:
                B = current_leadfield[:, np.newaxis]
            else:
                B = np.hstack([B, current_leadfield[:, np.newaxis]])

            # B = B / np.linalg.norm(B, axis=0)
            Q = I - B @ np.linalg.pinv(B)
            # Q -= Q.mean(axis=0)
            C = Q @ Us

            U, D, _= np.linalg.svd(C, full_matrices=False)
            # U -= U.mean(axis=0)
            
            # Truncate eigenvectors
            if truncate:
                Us = U[:, :n_comp-i]
            else:
                Us = U[:, :n_comp]
            

        # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
        source_covariance = csr_matrix(np.diag(source_covariance))
        L_s = self.leadfield @ source_covariance
        L = self.leadfield
        W = np.diag(np.linalg.norm(L, axis=0)) 
        # print(source_covariance.shape, L.shape, W.shape)
        inverse_operator = source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T

        return inverse_operator

    def prepare_flex(self):
        ''' Create the dictionary of increasingly smooth sources unless self.n_orders==0.
        
        Parameters
        ----------

        
        '''
        n_dipoles = self.leadfield.shape[1]
        self.identity = np.identity(n_dipoles)
        self.adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0).toarray()
        self.adjacency_ex = deepcopy(self.adjacency)
        np.fill_diagonal(self.adjacency_ex, 0)
        self.members_ex = [np.where(row)[0] for row in self.adjacency_ex]
        pos = pos_from_forward(self.forward, verbose=0)
        self.distances = cdist(pos, pos)
        self.is_prepared = True

    @staticmethod
    def get_comps_L(D):
        # L-curve method
        iters = np.arange(len(D))
        n_comp_L = find_corner(deepcopy(iters), deepcopy(D))
        return n_comp_L
    @staticmethod
    def get_comps_drop(D):
        D_ = D/D.max()
        n_comp_drop = np.where( abs(np.diff(D_)) < 0.001 )[0]

        if len(n_comp_drop) > 0:
            n_comp_drop = n_comp_drop[0] + 1
        else:
            n_comp_drop = 1
        return n_comp_drop         


# class SolverRAPMUSIC(BaseSolver):
#     ''' Class for the Recursively Applied Multiple Signal Classification
#     (RAP-MUSIC) inverse solution [1].
    
#     Attributes
#     ----------
    
#     References
#     ----------
#     [1] Mosher, J. C., & Leahy, R. M. (1999). Source localization using
#     recursively applied and projected (RAP) MUSIC. IEEE Transactions on signal
#     processing, 47(2), 332-340.
#     '''
#     def __init__(self, name="RAP-MUSIC", **kwargs):
#         self.name = name
#         return super().__init__(**kwargs)

#     def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto",  n="auto", k="auto", stop_crit=0.95, **kwargs):
#         ''' Calculate inverse operator.

#         Parameters
#         ----------
#         forward : mne.Forward
#             The mne-python Forward model instance.
#         mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
#             The MNE data object.
#         alpha : float
#             The regularization parameter.
#         n : int/ str
#             Number of eigenvectors to use or "auto" for l-curve method.
#         k : int
#             Number of recursions.
#         stop_crit : float
#             Criterion to stop recursions. The lower, the more dipoles will be
#             incorporated.
        
#         Return
#         ------
#         self : object returns itself for convenience
#         '''
#         super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
#         data = self.unpack_data_obj(mne_obj)

#         inverse_operator = self.make_rap(data, n, k, stop_crit)

#         self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
#         return self


#     def make_rap(self, y, n, k, stop_crit):
#         ''' Apply the RAP-MUSIC inverse solution to the EEG data.
        
#         Parameters
#         ----------
#         y : numpy.ndarray
#             EEG data matrix (channels, time)
#         n : int
#             Number of eigenvectors to use.
#         k : int
#             Number of recursions.
#         stop_crit : float
#             Criterion to stop recursions. The lower, the more
#             dipoles will be incorporated.

#         Return
#         ------
#         x_hat : numpy.ndarray
#             Source data matrix (sources, time)
#         '''
#         n_chans, n_dipoles = self.leadfield.shape
#         n_time = y.shape[1]
        
#         leadfield = self.leadfield
#         leadfield -= leadfield.mean(axis=0)
        
#         if k == "auto":
#             k = n_chans

#         # Data Covariance
#         C = y@y.T
#         I = np.identity(n_chans)
#         Q = np.identity(n_chans)

#         U, D, _= np.linalg.svd(C, full_matrices=False)
#         if n == "auto":
#             # L-curve method
#             iters = np.arange(len(D))
#             n_comp_L = find_corner(deepcopy(iters), deepcopy(D))
            
#             # Based on eigenvalue drop-off
#             D_ = D/D.max()
#             n_comp_drop = np.where( abs(np.diff(D_)) < 0.001 )[0]
#             if len(n_comp_drop) > 0:
#                 n_comp_drop = n_comp_drop[0] + 1
#             else:
#                 n_comp_drop = n_comp_L
            
#             # Combine the two:
#             n_comp = np.ceil((n_comp_drop + n_comp_L)/2).astype(int)

#             # import matplotlib.pyplot as plt
#             # plt.figure()
#             # plt.plot(iters, D_, '*k')
#             # plt.plot(iters[n_comp_drop], D_[n_comp_drop], 'og', label=f"Eig drop-off {n_comp_drop}")
#             # plt.plot(iters[n_comp_L], D_[n_comp_L], 'ob', label=f"L Curve Method {n_comp_L}")
#             # plt.plot(iters[n_comp], D_[n_comp], 'or', label=f"Combined {n_comp}")
#             # plt.legend()
#             # n_comp = n_comp_L

#         else:
#             n_comp = deepcopy(n)
#         Us = U[:, :n_comp]
#         dipole_idc = []
#         n_time = y.shape[1]
#         for i in range(k):
#             # print(i)
            
#             Ps = Us@Us.T

#             mu = np.zeros(n_dipoles)
#             for p in range(n_dipoles):
#                 l = leadfield[:, p][:, np.newaxis]
#                 norm_1 = np.linalg.norm(Ps @ Q @ l)
#                 norm_2 = np.linalg.norm(Q @ l)
                
#                 mu[p] = norm_1 / norm_2
            
#             if np.max(mu) < stop_crit:
#                 # print("breaking")
#                 break

#             dipole_idx = np.argmax(mu)
#             dipole_idc.append( dipole_idx )

#             if i == 0:
#                 B = leadfield[:, dipole_idx][:, np.newaxis]
#             else:
#                 B = np.hstack([B, leadfield[:, dipole_idx][:, np.newaxis]])
            
#             Q = I - B @ np.linalg.pinv(B)
#             C = Q @ Us

#             U, D, _= np.linalg.svd(C, full_matrices=False)
#             Us = U[:, :n_comp]

#         dipole_idc = np.array(dipole_idc).astype(int)
#         # x_hat = np.zeros((n_dipoles, n_time))
#         # x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
#         # return x_hat

#         # # WMNE-based
#         # # x_hat = np.zeros((n_dipoles, n_time))
#         # inverse_operator = np.zeros((n_dipoles, n_chans))
#         # L = self.leadfield[:, dipole_idc]
#         # W = np.diag(np.linalg.norm(L, axis=0))
#         # # x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y
#         # inverse_operator[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T

#         # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
#         source_covariance = np.zeros(n_dipoles)
#         source_covariance[dipole_idc] = 1
#         source_covariance = csr_matrix(np.diag(source_covariance))
#         L_s = self.leadfield @ source_covariance
#         L = self.leadfield
#         W = np.diag(np.linalg.norm(L, axis=0)) 
#         # print(source_covariance.shape, L.shape, W.shape)
#         inverse_operator = source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T


#         return inverse_operator

# class SolverTRAPMUSIC(BaseSolver):
#     ''' Class for the Truncated Recursively Applied Multiple Signal
#         Classification (TRAP-MUSIC) inverse solution [1].
    
#     Attributes
#     ----------
    
#     References
#     ----------
#     [1] Mäkelä, N., Stenroos, M., Sarvas, J., & Ilmoniemi, R. J. (2018).
#     Truncated rap-music (trap-music) for MEG and EEG source localization.
#     NeuroImage, 167, 73-83.
#     '''
#     def __init__(self, name="TRAP-MUSIC", **kwargs):
#         self.name = name
#         return super().__init__(**kwargs)

#     def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", n="auto", k="auto", stop_crit=0.95, **kwargs):
#         ''' Calculate inverse operator.

#         Parameters
#         ----------
#         forward : mne.Forward
#             The mne-python Forward model instance.
#         mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
#             The MNE data object.
#         alpha : float
#             The regularization parameter.
#         n : ["auto", int]
#             Number of eigenvectors to use.
#         k : int
#             Number of recursions.
#         stop_crit : float
#             Controls the percentage of top active dipoles that are selected
#             (i.e., sparsity).

#         Return
#         ------
#         self : object returns itself for convenience
#         '''

#         super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
#         data = self.unpack_data_obj(mne_obj)

#         inverse_operator = self.make_trap(data, n, k, stop_crit)

#         self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
#         return self

  

#     def make_trap(self, y, n, k, stop_crit):
#         ''' Apply the TRAP-MUSIC inverse solution to the EEG data.
        
#         Parameters
#         ----------
#         y : numpy.ndarray
#             EEG data matrix (channels, time)
#         n : int
#             Number of eigenvectors to use.
#         k : int
#             Number of recursions.
#         stop_crit : float
#             Criterion to stop recursions. The lower, the more
#             dipoles will be incorporated.

#         Return
#         ------
#         x_hat : numpy.ndarray
#             Source data matrix (sources, time)
#         '''
        
#         n_chans, n_dipoles = self.leadfield.shape
#         n_time = y.shape[1]
        
#         leadfield = self.leadfield
#         leadfield -= leadfield.mean(axis=0)
        
#         if k == "auto":
#             k = n_chans
        
#         # Data Covariance
#         C = y@y.T
#         I = np.identity(n_chans)
#         Q = np.identity(n_chans)
#         U, D, _= np.linalg.svd(C, full_matrices=False)
#         if n == "auto":
#             # L-curve method
#             iters = np.arange(len(D))
#             n_comp_L = find_corner(deepcopy(iters), deepcopy(D))
            
#             # Based on eigenvalue drop-off
#             D_ = D/D.max()
#             n_comp_drop = np.where( abs(np.diff(D_)) < 0.001 )[0]
#             if len(n_comp_drop) > 0:
#                 n_comp_drop = n_comp_drop[0] + 1
#             else:
#                 n_comp_drop = n_comp_L
            
#             # Combine the two:
#             n_comp = np.ceil((n_comp_drop + n_comp_L)/2).astype(int)

#             # import matplotlib.pyplot as plt
#             # plt.figure()
#             # plt.plot(iters, D_, '*k')
#             # plt.plot(iters[n_comp_drop], D_[n_comp_drop], 'og', label=f"Eig drop-off {n_comp_drop}")
#             # plt.plot(iters[n_comp_L], D_[n_comp_L], 'ob', label=f"L Curve Method {n_comp_L}")
#             # plt.plot(iters[n_comp], D_[n_comp], 'or', label=f"Combined {n_comp}")
#             # plt.legend()
#             # n_comp = n_comp_L

#         else:
#             n_comp = deepcopy(n)
#         Us = U[:, :n_comp]
#         dipole_idc = []
#         n_time = y.shape[1]
#         for i in range(k):
#             # print(i)
            
#             Ps = Us@Us.T

#             mu = np.zeros(n_dipoles)
#             for p in range(n_dipoles):
#                 l = leadfield[:, p][:, np.newaxis]
#                 norm_1 = np.linalg.norm(Ps @ Q @ l)
#                 norm_2 = np.linalg.norm(Q @ l)
#                 mu[p] = norm_1 / norm_2
            
#             if np.max(mu) < stop_crit:
#                 # print("breaking")
#                 break

#             dipole_idx = np.argmax(mu)
#             dipole_idc.append( dipole_idx )

#             if i == 0:
#                 B = leadfield[:, dipole_idx][:, np.newaxis]
#             else:
#                 B = np.hstack([B, leadfield[:, dipole_idx][:, np.newaxis]])
            
#             Q = I - B @ np.linalg.pinv(B)
#             C = Q @ Us
#             U, D, _= np.linalg.svd(C, full_matrices=False)
#             Us = U[:, :n_comp-i]
#         dipole_idc = np.array(dipole_idc).astype(int)
        
#         # x_hat = np.zeros((n_dipoles, n_time))

#         # x_hat[dipole_idc, :] = np.linalg.pinv(leadfield[:, dipole_idc]) @ y
#         # return x_hat

#         # # WMNE-based
#         # x_hat = np.zeros((n_dipoles, n_time))
#         # inverse_operator = np.zeros((n_dipoles, n_chans))
#         # L = self.leadfield[:, dipole_idc]
#         # W = np.diag(np.linalg.norm(L, axis=0))
#         # x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y
#         # inverse_operator[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T

#         # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
#         source_covariance = np.zeros(n_dipoles)
#         source_covariance[dipole_idc] = 1
#         source_covariance = csr_matrix(np.diag(source_covariance))
#         L_s = self.leadfield @ source_covariance
#         L = self.leadfield
#         W = np.diag(np.linalg.norm(L, axis=0)) 
#         # print(source_covariance.shape, L.shape, W.shape)
#         inverse_operator = source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T



#         return inverse_operator
