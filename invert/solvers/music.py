from time import time
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
        # leadfield -= leadfield.mean(axis=0)
        
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
        estimation (FLEX-MUSIC, [1]).
    
    Attributes
    ----------
    
    References
    ---------
    [1] Hecker, L., Tebartz van Elst, L., & Kornmeier, J. (2023). Source
    localization using recursively applied and projected MUSIC with flexible
    extent estimation. Frontiers in Neuroscience, 17, 1170862.
    
    '''
    def __init__(self, name="FLEX-MUSIC", scale_leadfield=False, **kwargs):
        self.name = name
        self.is_prepared = False
        self.scale_leadfield = scale_leadfield
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, n_orders=3, 
                              truncate=False, alpha="auto", n="auto", 
                              k="auto", stop_crit=0.95, refine_solution=False, 
                              max_iter=1000, diffusion_smoothing=True, 
                              diffusion_parameter=0.1, adjacency_type="spatial", 
                              adjacency_distance=3e-3, depth_weights=None,**kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        n_orders : int
            Controls the maximum smoothness to pursue.
        truncate : bool
            If True: Truncate SVD's eigenvectors (like TRAP-MUSIC), otherwise
            don't (like RAP-MUSIC).
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
        diffusion_smoothing : bool
            Whether to use diffusion smoothing. Default is True.
        diffusion_parameter : float
            The diffusion parameter (alpha). Default is 0.1.
        adjacency_type : str
            The type of adjacency. "spatial" -> based on graph neighbors. "distance" -> based on distance
        adjacency_distance : float
            The distance at which neighboring dipoles are considered neighbors.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        
        self.diffusion_smoothing = diffusion_smoothing, 
        self.diffusion_parameter = diffusion_parameter
        self.n_orders = n_orders
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        self.truncate = truncate
        
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()
        
        inverse_operator = self.make_flex(data, n, k, stop_crit, 
                                          truncate, refine_solution=refine_solution, 
                                          max_iter=max_iter,
                                          depth_weights=depth_weights)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_flex(self, y, n, k, stop_crit, truncate, refine_solution=False, max_iter=1000, depth_weights=None):
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
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''

        n_chans, n_dipoles = self.leadfield.shape
        n_time = y.shape[1]

        leadfield = self.leadfield
        # leadfield -= leadfield.mean(axis=0)
        
        leadfields = self.leadfields
        n_orders = len(self.leadfields)
        if k == "auto":
            k = n_chans
        # Assert common average reference
        # y -= y.mean(axis=0)
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
                n_comp_mean = np.where(D<D.mean())[0][0]

                # Combine the two:
                n_comp = np.ceil(1.1*np.mean([n_comp_L, n_comp_drop, n_comp_mean])).astype(int)
        else:
            n_comp = deepcopy(n)
        # print("n_comp: ", n_comp)
        Us = U[:, :n_comp]
        
        C_initial = Us @ Us.T
        dipole_idc = []
        source_covariance = np.zeros(n_dipoles)
        S_AP = []
        for i in range(k):
            print(f"Source Iteration ", i)
            Ps = Us @ Us.T
            PsQ = Ps @ Q

            mu = np.zeros((n_orders, n_dipoles))
            for nn in range(n_orders):
                norm_1 = np.linalg.norm(PsQ @ leadfields[nn], axis=0)
                norm_2 = np.linalg.norm(Q @ leadfields[nn], axis=0) 
                # norm_1 = np.diag(leadfields[nn].T @ PsQ @ leadfields[nn])
                # norm_2 = np.diag(leadfields[nn].T @ Q @ leadfields[nn])
                mu[nn, :] = norm_1 / norm_2

                
            self.mu = mu
            # Find the dipole/ patch with highest correlation with the residual
            best_order, best_dipole = np.unravel_index(np.argmax(mu), mu.shape)
            
            if i>0 and np.max(mu) < stop_crit:
                print(f"\t break because mu is ", np.max(mu))
                break
            S_AP.append([best_order, best_dipole])

            # source_covariance += np.squeeze(self.gradients[best_order][best_dipole] * (1/np.sqrt(i+1)))
            # source_covariance += np.squeeze(self.gradients[best_order][best_dipole])

            if i == 0:
                B = leadfields[best_order][:, best_dipole][:, np.newaxis]
            else:
                B = np.hstack([B, leadfields[best_order][:, best_dipole][:, np.newaxis]])

            Q = I - B @ np.linalg.pinv(B)
            C = Q @ Us
            U, D, _= np.linalg.svd(C, full_matrices=False)
            
            # Truncate eigenvectors
            if truncate:
                Us = U[:, :n_comp-i]
            else:
                Us = U[:, :n_comp]
        # Phase 2: refinement
        C = C_initial
        S_AP_2 = deepcopy(S_AP)
        print(S_AP)
        
        if len(S_AP) > 1 and refine_solution:
            # best_vals = np.zeros(n_comp)
            for iter in range(max_iter):
                S_AP_2_Prev = deepcopy(S_AP_2)
                for q in range(len(S_AP)):
                    S_AP_TMP = S_AP_2.copy()
                    S_AP_TMP.pop(q)
                    
                    B = np.stack([leadfields[order][:, dipole] for order, dipole in S_AP_TMP], axis=1)

                    # Q = I - B @ np.linalg.pinv(B)
                    # Ps = C_initial

                    P_A = B @ np.linalg.pinv(B.T @ B) @ B.T
                    Q = np.identity(P_A.shape[0]) - P_A

                    ap_val2 = np.zeros((n_orders, n_dipoles))
                    for nn in range(n_orders):
                        L = leadfields[nn]
                        upper = np.diag(L.T @ Q @ C @ Q @ L)
                        lower = np.diag(L.T @ Q @ L)
                        # upper = np.linalg.norm(Ps @ Q @ L, axis=0)
                        # lower = np.linalg.norm(Q @ L, axis=0) 
                        ap_val2[nn] = upper / lower
                    
                    best_order, best_dipole = np.unravel_index(np.argmax(ap_val2), ap_val2.shape)
                    # best_val = ap_val2.max()
                    S_AP_2[q] = [best_order, best_dipole]
                    # print(f"refinement: adding new value {best_val} at idx {best_dipole}, best_order {best_order}")
                    # best_vals[q] = best_val

                if iter > 0 and S_AP_2 == S_AP_2_Prev:
                    break

        self.candidates = S_AP_2
        source_covariance = np.sum([np.squeeze(self.gradients[order][dipole].toarray()) for order, dipole in S_AP_2], axis=0)
        
        # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
        nonzero = np.where(source_covariance!=0)[0]
        inverse_operator = np.zeros((n_dipoles, n_chans))

        source_covariance = csr_matrix(np.diag(source_covariance[nonzero]))
        L = self.leadfield[:, nonzero]
        if depth_weights is not None:
            L = L * depth_weights[nonzero]
        
        # Version 7: Standard Minimum Norm Estimate with Source Covariance (MNE PDF p.122)
        # alpha = np.trace(L.T @ L) / np.trace(L @ source_covariance @ L.T) #
        # source_covariance *= alpha
        # inverse_operator[nonzero, :] = source_covariance @ L.T @ np.linalg.pinv(L @ source_covariance @ L.T)

        # Version 8: Lower rank MNE
        n = len(S_AP_2)
        source_covariance = np.identity(n)
        L = np.stack([leadfields[order][:, dipole] for order, dipole in S_AP_2], axis=1)
        gradients = np.squeeze(np.stack([self.gradients[order][dipole].toarray() for order, dipole in S_AP_2], axis=1))
        inverse_operator = gradients.T @ source_covariance @ L.T @ np.linalg.pinv(L @ source_covariance @ L.T)
        
        return inverse_operator

    # def prepare_flex(self):
    #     ''' Create the dictionary of increasingly smooth sources unless self.n_orders==0.
        
    #     Parameters
    #     ----------

    #     '''
    #     n_dipoles = self.leadfield.shape[1]
    #     I = np.identity(n_dipoles)

    #     adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        
    #     LL = laplacian(adjacency)
    #     self.leadfields = [deepcopy(self.leadfield), ]
    #     self.gradients = [csr_matrix(I),]
        
    #     if self.diffusion_smoothing:
    #         smoothing_operator = csr_matrix(I - self.diffusion_parameter * LL)
    #     else:
    #         smoothing_operator = csr_matrix(abs(LL))

    #     for _ in range(self.n_orders):
    #         new_leadfield = self.leadfields[-1] @ smoothing_operator
    #         new_gradient = self.gradients[-1] @ smoothing_operator
        
    #         self.leadfields.append( new_leadfield )
    #         self.gradients.append( new_gradient )
            
    #     # scale and transform gradients
    #     for i in range(self.n_orders+1):
    #         # self.gradients[i] =self.gradients[i].toarray() / self.gradients[i].toarray().max(axis=0)
    #         row_max = self.gradients[i].max(axis=1).toarray().ravel()
    #         scaling_factors = 1 / row_max
    #         self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))
            
    #     self.is_prepared = True

    def prepare_flex(self):
        ''' Create the dictionary of increasingly smooth sources unless
        self.n_orders==0. Flexibly selects diffusion parameter, too.
        
        Parameters
        ----------

        '''
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)
        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(self.forward['src'], self.adjacency_distance, verbose=None)
        
        LL = laplacian(adjacency)
        self.leadfields = [deepcopy(self.leadfield), ]
        self.gradients = [csr_matrix(I),]
        

        # if self.diffusion_smoothing:
        #     smoothing_operator = csr_matrix(I - self.diffusion_parameter * LL)
        # else:
        #     smoothing_operator = csr_matrix(abs(LL))
        if self.diffusion_parameter == "auto":
            alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            smoothing_operators = [csr_matrix(I - alpha * LL) for alpha in alphas]
        else:
            smoothing_operators = [csr_matrix(I - self.diffusion_parameter * LL),]


        for smoothing_operator in smoothing_operators:

            for i in range(self.n_orders):
                smoothing_operator_i = smoothing_operator**(i+1)  # csr_matrix(np.linalg.matrix_power(smoothing_operator.toarray(), i+1))
                new_leadfield = self.leadfields[0] @ smoothing_operator_i
                new_gradient = self.gradients[0] @ smoothing_operator_i

                # Scaling? Not sure...
                if self.scale_leadfield:
                    # new_leadfield -= new_leadfield.mean(axis=0)
                    new_leadfield /= np.linalg.norm(new_leadfield, axis=0)
            
                self.leadfields.append( new_leadfield )
                self.gradients.append( new_gradient )
        # scale and transform gradients
        for i in range(len(self.gradients)):
            # self.gradients[i] = self.gradients[i].toarray() / self.gradients[i].toarray().max(axis=0)
            # row_max = self.gradients[i].max(axis=1).toarray().ravel()
            # scaling_factors = 1 / row_max
            row_sums = self.gradients[i].sum(axis=1).ravel()  # Compute the sum of each row
            scaling_factors = 1 / row_sums
            self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))
        
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

class SolverAlternatingProjections(BaseSolver):
    ''' Class for the Alternating Projections inverse solution [1] with flexible
        extent estimation (FLEX-AP). This approach combines the AP-approach by
        Adler et al. [1] with dipoles with flexible extents, e.g., FLEX-MUSIC
        (Hecker 2023, unpublished).
    
    Attributes
    ----------
    n_orders : int
        Controls the maximum smoothness to pursue.
    
    References
    ---------
    [1] Adler, A., Wax, M., & Pantazis, D. (2022, March). Brain Source
    Localization by Alternating Projection. In 2022 IEEE 19th International
    Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE.

    '''
    def __init__(self, name="Flexible Alternative Projections", scale_leadfield=False, **kwargs):
        self.name = name
        self.is_prepared = False
        self.scale_leadfield = scale_leadfield
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, n_orders=3, 
                              alpha="auto", n="auto", k="auto", stop_crit=0.95,
                              refine_solution=True, max_iter=1000, diffusion_smoothing=True, 
                              diffusion_parameter=0.1, adjacency_type="spatial", 
                              adjacency_distance=3e-3, depth_weights=None, **kwargs):
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
        max_iter : int
            Maximum number of iterations during refinement.
        diffusion_smoothing : bool
            Whether to use diffusion smoothing. Default is True.
        diffusion_parameter : float
            The diffusion parameter (alpha). Default is 0.1.
        adjacency_type : str
            The type of adjacency. "spatial" -> based on graph neighbors. "distance" -> based on distance
        adjacency_distance : float
            The distance at which neighboring dipoles are considered neighbors.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.

        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        self.diffusion_smoothing = diffusion_smoothing
        self.diffusion_parameter = diffusion_parameter
        self.n_orders = n_orders
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()
        inverse_operator = self.make_ap(data, n, k, 
                                        max_iter=max_iter, 
                                        refine_solution=refine_solution,
                                        depth_weights=depth_weights)
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_ap(self, y, n, k, refine_solution=True, max_iter=1000, covariance_type="AP", depth_weights=None, apply_depth_weights=True):
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
        refine_solution : bool
            If True: Re-visit each selected candidate and check if there is a
            better alternative.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.
        apply_depth_weights : bool
            Whether to apply depth weights to the leadfields.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        n_chans, n_dipoles = self.leadfield.shape
        n_time = y.shape[1]

        leadfield = self.leadfield
        
        leadfields = self.leadfields
        n_orders = len(self.leadfields)
        if k == "auto":
            k = n_chans        
        
        # Estimate number of components
        if type(n) == str:
            C = y @ y.T
            U, D, _= np.linalg.svd(C, full_matrices=False)
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
                n_comp_mean = np.where(D<D.mean())[0][0]

                # Combine the two and bias it slightly:
                n_comp = np.ceil(1.1*np.mean([n_comp_L, n_comp_drop, n_comp_mean])).astype(int)
        else:
            n_comp = deepcopy(n)
        
        # MUSIC TYPE
        if covariance_type == "MUSIC":
            C = y@y.T
            Q = np.identity(n_chans)
            U, D, _= np.linalg.svd(C, full_matrices=False)
            
            Us = U[:, :n_comp]
            
            # MUSIC subspace-based Covariance
            C = Us @ Us.T

        elif covariance_type == "AP":  # Normal covariance
            mu = 0  # 1e-3
            C = y@y.T + mu * np.trace(np.matmul(y,y.T)) * np.eye(y.shape[0]) # Array Covariance matrix
        else:
            msg = f"covariance_type must be MUSIC or AP but is {covariance_type}"
            raise AttributeError(msg)

        
        S_AP = []
        # Initialization:  search the 1st source location over the entire
        # dipoles topographies space
        ap_val1 = np.zeros((n_orders, n_dipoles))
        for nn in range(n_orders):
            L = leadfields[nn]
            # norm_1 = np.diag(L.T @ C @ L)
            norm_1 = np.einsum('ij,jk,ki->i', L.T, C, L)
            norm_2 = np.diag(L.T @ L) # not necessary since leadfields were L2-normalized before
            ap_val1[nn, :] = norm_1 / norm_2
        self.mu = ap_val1
        best_order, best_dipole = np.unravel_index(np.argmax(ap_val1), ap_val1.shape)
        S_AP.append( [best_order, best_dipole] )
        
        # (b) Now, add one source at a time
        for ii in range(1, n_comp):
            ap_val2 = np.zeros((n_orders, n_dipoles))
            # Compose current leadfield components
            A = np.stack([leadfields[order][:, dipole] for order, dipole in S_AP], axis=1)
            P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
            Q = np.identity(P_A.shape[0]) - P_A
            # QCQ = Q @ C @ Q
            # QCQ = np.dot(Q, C).dot(Q)
            QC = Q @ C
            for nn in range(n_orders):
                L = leadfields[nn]
                # QL = np.dot(Q, L)
                # ap_val2[nn] = np.sum(L * QCQ.dot(L), axis=0) / np.sum(L * QL, axis=0)  # fast, but unstable
                ap_val2[nn] = np.sum(L * (QC @ (Q @ L)), axis=0) / np.sum(L * (Q @ L), axis=0)  # fast and stable

                # Old, slow
                # upper = np.diag(L.T @ QCQ @ L)
                # lower = np.diag(L.T @ Q @ L)
                # ap_val2[nn] = upper / lower
            
            # Select the best candidate, unless it is already in the set
            # best_order, best_dipole = np.unravel_index(np.argmax(ap_val2), ap_val2.shape)
            # S_AP.append( [best_order, best_dipole] )
            select_idx = -1
            while True:
                best_order, best_dipole = np.unravel_index(np.argsort(ap_val2.flatten())[select_idx], ap_val2.shape)
                candidate = [best_order, best_dipole]
                if not candidate in S_AP:
                    S_AP.append( [best_order, best_dipole] )
                    break
                print("rerolled AP candidate")
                select_idx -= 1
                
            if len(S_AP) != len(set(tuple(row) for row in S_AP)):
                print("Found duplicate candidates in AP!!!!")
        
        # Phase 2: refinement
        S_AP_2 = deepcopy(S_AP)
        if len(S_AP) > 1 and refine_solution:
            # best_vals = np.zeros(n_comp)
            for iter in range(max_iter):
                S_AP_2_Prev = deepcopy(S_AP_2)
                for q in range(len(S_AP)):
                    S_AP_TMP = S_AP_2.copy()
                    S_AP_TMP.pop(q)
                    
                    A = np.stack([leadfields[order][:, dipole] for order, dipole in S_AP_TMP], axis=1)
                    P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
                    Q = np.identity(P_A.shape[0]) - P_A
                    # QCQ = np.dot(Q, C).dot(Q)
                    QC = Q @ C
                    ap_val2 = np.zeros((n_orders, n_dipoles))
                    for nn in range(n_orders):
                        # New, fast
                        L = leadfields[nn]
                        # QL = np.dot(Q, L)
                        
                        # ap_val2[nn] = np.sum(L * QCQ.dot(L), axis=0) / np.sum(L * QL, axis=0)  # fast, but unstable
                        ap_val2[nn] = np.sum(L * (QC @ (Q @ L)), axis=0) / np.sum(L * (Q @ L), axis=0)  # fast and stable

                        # Old, slow
                        # upper = np.diag(L.T @ QCQ @ L)
                        # lower = np.diag(L.T @ Q @ L)
                        # ap_val2[nn] = upper / lower
                    
                    # Select the best candidate, unless it is already in the set
                    # best_order, best_dipole = np.unravel_index(np.argmax(ap_val2), ap_val2.shape)
                    # S_AP_2[q] = [best_order, best_dipole]
                    select_idx = -1
                    while True:
                        best_order, best_dipole = np.unravel_index(np.argsort(ap_val2.flatten())[select_idx], ap_val2.shape)
                        candidate = [best_order, best_dipole]
                        if not candidate in S_AP_2[:q] and not candidate in S_AP_2[q+1:]:
                            S_AP_2[q] = [best_order, best_dipole]
                            break
                        print("rerolled AP candidate")
                        select_idx -= 1

                    
                    # S_AP_2[q] = [best_order, best_dipole]
                    if len(S_AP_2) != len(set(tuple(row) for row in S_AP_2)):
                        print("Found duplicate candidates in AP refinement")
                    # print(f"refinement: adding new value {best_val} at idx {best_dipole}, best_order {best_order}")
                    # best_vals[q] = best_val
                    
                if S_AP_2 == S_AP_2_Prev:  # and iter>0:
                    break
        self.candidates = S_AP_2
        source_covariance = np.sum([np.squeeze(self.gradients[order][dipole].toarray()) for order, dipole in S_AP_2], axis=0)

        # Prior-Cov based version 2: Use the selected smooth patches as source covariance priors
        nonzero = np.where(source_covariance!=0)[0]
        inverse_operator = np.zeros((n_dipoles, n_chans))
    	
        source_covariance = csr_matrix(np.diag(source_covariance[nonzero]))
        L = self.leadfield[:, nonzero]
        if depth_weights is not None:
            L = L * depth_weights[nonzero]
        
        # Version 8: Lower rank MNE
        n = len(S_AP_2)
        source_covariance = np.identity(n)
        L = np.stack([leadfields[order][:, dipole] for order, dipole in S_AP_2], axis=1)
        gradients = np.squeeze(np.stack([self.gradients[order][dipole].toarray() for order, dipole in S_AP_2], axis=1))
        # print(source_covariance.shape, L.shape, gradients.shape)
        inverse_operator = gradients.T @ source_covariance @ L.T @ np.linalg.pinv(L @ source_covariance @ L.T)
        # print(inverse_operator.shape)
        return inverse_operator

    def prepare_flex(self):
        ''' Create the dictionary of increasingly smooth sources unless
        self.n_orders==0. Flexibly selects diffusion parameter, too.
        
        Parameters
        ----------

        '''
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)
        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(self.forward['src'], self.adjacency_distance, verbose=None)
        
        LL = laplacian(adjacency)
        self.leadfields = [deepcopy(self.leadfield), ]
        self.gradients = [csr_matrix(I),]

        if self.diffusion_parameter == "auto":
            alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            smoothing_operators = [csr_matrix(I - alpha * LL) for alpha in alphas]
        else:
            smoothing_operators = [csr_matrix(I - self.diffusion_parameter * LL),]


        for smoothing_operator in smoothing_operators:

            for i in range(self.n_orders):
                smoothing_operator_i = smoothing_operator**(i+1)  # csr_matrix(np.linalg.matrix_power(smoothing_operator.toarray(), i+1))
                new_leadfield = self.leadfields[0] @ smoothing_operator_i
                new_gradient = self.gradients[0] @ smoothing_operator_i

                # Scaling? Not sure...
                if self.scale_leadfield:
                    # new_leadfield -= new_leadfield.mean(axis=0)
                    new_leadfield /= np.linalg.norm(new_leadfield, axis=0)
            
                self.leadfields.append( new_leadfield )
                self.gradients.append( new_gradient )
        # scale and transform gradients
        for i in range(len(self.gradients)):
            row_sums = self.gradients[i].sum(axis=1).ravel()  # Compute the sum of each row
            scaling_factors = 1 / row_sums
            self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))
        
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


class SolverSignalSubspaceMatching(BaseSolver):
    ''' Class for the Signal Subspace Matching inverse solution with flexible
        extent estimation (FLEX-AP). This approach combines the SSM-approach by
        Adler et al. [1] with dipoles with flexible extents, e.g., FLEX-MUSIC
        (Hecker 2023, unpublished, [2]).
    
    Attributes
    ----------
    n_orders : int
        Controls the maximum smoothness to pursue.
    
    References
    ---------
    [1] Wax, M., & Adler, A. (2021). Direction of arrival estimation in the
    presence of model errors by signal subspace matching. Signal Processing,
    181, 107900.
    [2] TBD.

    '''
    def __init__(self, name="Flexible Signal Subspace Matching", scale_leadfield=False, **kwargs):
        self.name = name
        self.is_prepared = False
        self.scale_leadfield = scale_leadfield
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, n_orders=3, 
                              alpha="auto", n="auto", refine_solution=True, 
                              max_iter=5, diffusion_smoothing=True, 
                              diffusion_parameter=0.1, adjacency_type="spatial", 
                              adjacency_distance=3e-3, lambda_reg1=0.001, 
                              lambda_reg2=0.0001, lambda_reg3=0.0, **kwargs):
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
        stop_crit : float
            Criterion to stop recursions. The lower, the more dipoles will be
            incorporated.
        max_iter : int
            Maximum number of iterations during refinement.
        diffusion_smoothing : bool
            Whether to use diffusion smoothing. Default is True.
        diffusion_parameter : float
            The diffusion parameter (alpha). Default is 0.1.
        adjacency_type : str
            The type of adjacency. "spatial" -> based on graph neighbors. "distance" -> based on distance
        adjacency_distance : float
            The distance at which neighboring dipoles are considered neighbors.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.

        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        self.diffusion_smoothing = diffusion_smoothing, 
        self.diffusion_parameter = diffusion_parameter
        self.n_orders = n_orders
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()
        inverse_operator = self.make_ssm(data, n, 
                                        max_iter=max_iter, 
                                        refine_solution=refine_solution,
                                        lambda_reg1=lambda_reg1,
                                        lambda_reg2=lambda_reg2,
                                        lambda_reg3=lambda_reg3)
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_ssm(self, Y, n, refine_solution=True, max_iter=5, 
                 lambda_reg1=0.001, lambda_reg2=0.0001, lambda_reg3=0.0):
        ''' Create the FLEX-SSM inverse solution to the EEG data.
        
        Parameters
        ----------
        Y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int/ str
            Number of eigenvectors to use or "auto" for l-curve method.
        refine_solution : bool
            If True: Re-visit each selected candidate and check if there is a
            better alternative.
        lambda_reg1 : float
            Regularization parameter for the projection matrix.
        lambda_reg2 : float
            Regularization parameter for the source covariance matrix.
        lambda_reg3 : float
            Regularization parameter for the source covariance matrix.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        n_chans, n_dipoles = self.leadfield.shape
        n_time = Y.shape[1]

        leadfields = self.leadfields
        n_orders = len(self.leadfields)
        
        # Determine the number of sources
        if type(n) == str:
            C = Y @ Y.T
            U, D, _= np.linalg.svd(C, full_matrices=False)
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
                n_comp_mean = np.where(D<D.mean())[0][0]

                # Combine the two and bias it slightly:
                n_comp = np.ceil(1.1*np.mean([n_comp_L, n_comp_drop, n_comp_mean])).astype(int)
        else:
            n_comp = deepcopy(n)
        
        M_Y = Y.T @ Y
        YY = M_Y + lambda_reg1 * np.trace(M_Y) * np.eye(n_time)
        P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
        P_A = np.zeros((n_chans, n_chans))

        S_SSM = []
        A_q = []
        
        # Initial source location
        S_SSM.append( self.get_source_ssm(P_Y, P_A, leadfields, lambda_reg=lambda_reg3) )
        # Now, add one source at a time
        for qq in range(1, n_comp):
            order, location = S_SSM[-1]
            A_q.append(leadfields[order][:, location])
            P_A = self.compute_projection_matrix(A_q, lambda_reg=lambda_reg2)
            S_SSM.append( self.get_source_ssm(P_Y, P_A, leadfields, S_SSM, lambda_reg=lambda_reg3) )
        
        A_q.append( leadfields[S_SSM[-1][0]][:, S_SSM[-1][1]] )
        
        # Phase 2: refinement
        S_SSM_2 = deepcopy(S_SSM)
        if len(S_SSM_2) > 1 and refine_solution:
            S_SSM_prev = deepcopy(S_SSM_2)
            for j in range(max_iter):
                A_q_j = A_q.copy()
                for qq in range(n_comp):
                    A_temp = np.delete(A_q_j, qq, axis=0) # delete the current source
                    qq_temp = np.delete(S_SSM_2, qq, axis=0) # delete the current source
                    P_A = self.compute_projection_matrix(A_temp, lambda_reg=lambda_reg2)
                    S_SSM_2[qq] = self.get_source_ssm(P_Y, P_A, leadfields, qq_temp, lambda_reg=lambda_reg3)
                    A_q_j[qq] = leadfields[S_SSM_2[qq][0]][:, S_SSM_2[qq][1]]
                    
                if S_SSM_2 == S_SSM_prev:
                    # print(f"No change after {j+1} iterations")
                    break
                # else:
                #     print(S_SSM_prev, " ==> ", S_SSM_2)
                S_SSM_prev = deepcopy(S_SSM_2)
        self.candidates = S_SSM_2       

        # Low-rank minimum norm solution
        source_covariance = np.identity(n_comp)
        L = np.stack([leadfields[order][:, dipole] for order, dipole in S_SSM_2], axis=1)
        gradients = np.squeeze(np.stack([self.gradients[order][dipole].toarray() for order, dipole in S_SSM_2], axis=1))
        inverse_operator = gradients.T @ source_covariance @ L.T @ np.linalg.pinv(L @ source_covariance @ L.T)

        return inverse_operator

    @staticmethod
    def get_source_ssm(P_Y: np.ndarray, P_A: np.ndarray, leadfields: list, q_ignore: list=[], lambda_reg=0.0):
        ''' Compute the source with the highest AP value.
        Parameters
        ----------
        P_Y : numpy.ndarray
            The projection matrix of the data covariance.
        P_A : numpy.ndarray
            The projection matrix of the source covariance.
        leadfields : list
            The list of leadfields.
        q_ignore : list
            List of sources to ignore (e.g., already selected sources).
        lambda_reg : float
            Regularization parameter to enhance stability. This is not within the original SSM algorithm.
        '''
        n_dipoles = leadfields[0].shape[1]
        n_orders = len(leadfields)

        C = P_Y.T @ P_Y
        R = np.eye(P_A.shape[0]) - P_A

        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            a_s = R @ leadfields[jj]

            # Fast
            upper = np.einsum('ij,ij->j', a_s, C @ a_s)
            lower = np.einsum('ij,ij->j', a_s, a_s) + lambda_reg
            expression[jj, :] = upper  / lower
            
            # Slow
            # upper = a_s.T @ C @ a_s
            # lower = a_s.T @ a_s + np.eye(n_dipoles) * lambda_reg
            # # print(upper.shape, lower.shape)
            # expression[jj, :] = np.diag(upper @ np.linalg.inv(lower))


        if q_ignore != []:
            for order, dipole in q_ignore:
                expression[order, dipole] = np.nan
        # print(f"Expression:\n{expression}")
        order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
        return order, dipole

    @staticmethod
    def compute_projection_matrix(A_q, lambda_reg=0.0001):
        ''' Compute the projection matrix for the SSM algorithm.'''
        A_q = np.stack(A_q, axis=1)
        M_A = A_q.T @ A_q
        AA = M_A + lambda_reg * np.trace(M_A) * np.eye(M_A.shape[0])
        P_A = (A_q @ np.linalg.inv(AA)) @ A_q.T
        return P_A

    def prepare_flex(self):
        ''' Create the dictionary of increasingly smooth sources unless
        self.n_orders==0. Flexibly selects diffusion parameter, too.
        
        Parameters
        ----------

        '''
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)
        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(self.forward['src'], self.adjacency_distance, verbose=None)
        
        LL = laplacian(adjacency)
        self.leadfields = [deepcopy(self.leadfield), ]
        self.gradients = [csr_matrix(I),]

        if self.diffusion_parameter == "auto":
            alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            smoothing_operators = [csr_matrix(I - alpha * LL) for alpha in alphas]
        else:
            smoothing_operators = [csr_matrix(I - self.diffusion_parameter * LL),]


        for smoothing_operator in smoothing_operators:

            for i in range(self.n_orders):
                smoothing_operator_i = smoothing_operator**(i+1)  # csr_matrix(np.linalg.matrix_power(smoothing_operator.toarray(), i+1))
                new_leadfield = self.leadfields[0] @ smoothing_operator_i
                new_gradient = self.gradients[0] @ smoothing_operator_i

                # Scaling? Not sure...
                if self.scale_leadfield:
                    # new_leadfield -= new_leadfield.mean(axis=0)
                    new_leadfield /= np.linalg.norm(new_leadfield, axis=0)
            
                self.leadfields.append( new_leadfield )
                self.gradients.append( new_gradient )
        # scale and transform gradients
        for i in range(len(self.gradients)):
            row_sums = self.gradients[i].sum(axis=1).ravel()  # Compute the sum of each row
            scaling_factors = 1 / row_sums
            self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))
        
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


class SolverAdaptiveAlternatingProjections(BaseSolver):
    ''' Class for the Alternating Projections inverse solution [1] with flexible
        extent estimation (FLEX-AP). This approach combines the AP-approach by
        Adler et al. [1] with dipoles with flexible extents, e.g., FLEX-MUSIC
        (Hecker 2023, unpublished).
    
    Attributes
    ----------
    n_orders : int
        Controls the maximum smoothness to pursue.
    
    References
    ---------
    [1] Adler, A., Wax, M., & Pantazis, D. (2022, March). Brain Source
    Localization by Alternating Projection. In 2022 IEEE 19th International
    Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE.

    '''
    def __init__(self, name="Flexible Alternative Projections", scale_leadfield=False, **kwargs):
        self.name = name
        self.is_prepared = False
        self.scale_leadfield = scale_leadfield
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, n_orders=3, 
                              alpha="auto", n="auto", k="auto", stop_crit=0.95,
                              refine_solution=True, max_iter=1000, diffusion_smoothing=True, 
                              diffusion_parameter=0.1, adjacency_type="spatial", 
                              adjacency_distance=3e-3, depth_weights=None, **kwargs):
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
        max_iter : int
            Maximum number of iterations during refinement.
        diffusion_smoothing : bool
            Whether to use diffusion smoothing. Default is True.
        diffusion_parameter : float
            The diffusion parameter (alpha). Default is 0.1.
        adjacency_type : str
            The type of adjacency. "spatial" -> based on graph neighbors. "distance" -> based on distance
        adjacency_distance : float
            The distance at which neighboring dipoles are considered neighbors.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.

        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        self.diffusion_smoothing = diffusion_smoothing, 
        self.diffusion_parameter = diffusion_parameter
        self.n_orders = n_orders
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()
        inverse_operator = self.make_ap(data, n, k, 
                                        max_iter=max_iter, 
                                        refine_solution=refine_solution,
                                        depth_weights=depth_weights)
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_ap(self, y, n, k, refine_solution=True, max_iter=1000, covariance_type="AP", depth_weights=None, apply_depth_weights=True):
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
        refine_solution : bool
            If True: Re-visit each selected candidate and check if there is a
            better alternative.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.
        apply_depth_weights : bool
            Whether to apply depth weights to the leadfields.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        n_chans, n_dipoles = self.leadfield.shape
        n_time = y.shape[1]

        leadfield = self.leadfield
        # leadfield -= leadfield.mean(axis=0)
        
        leadfields = self.leadfields
        n_orders = len(self.leadfields)
        if k == "auto":
            k = n_chans
        # Assert common average reference
        # y -= y.mean(axis=0)
        max_rank = 4
        min_change = 1.001
        
        
        # Compute Data Covariance
        if type(n) == str:
            C = y@y.T
            U, D, _= np.linalg.svd(C, full_matrices=False)
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
                n_comp_mean = np.where(D<D.mean())[0][0]

                # Combine the two and bias it slightly:
                n_comp = np.ceil(1.1*np.mean([n_comp_L, n_comp_drop, n_comp_mean])).astype(int)
        else:
            n_comp = deepcopy(n)
        
        # MUSIC TYPE
        if covariance_type == "MUSIC":
            C = y@y.T
            Q = np.identity(n_chans)
            U, D, _= np.linalg.svd(C, full_matrices=False)
            
            Us = U[:, :n_comp]
            
            # MUSIC subspace-based Covariance
            C = Us @ Us.T

        elif covariance_type == "AP":  # Normal covariance
            mu = 0  # 1e-3
            C = y@y.T + mu * np.trace(np.matmul(y,y.T)) * np.eye(y.shape[0]) # Array Covariance matrix
        else:
            msg = f"covariance_type must be MUSIC or AP but is {covariance_type}"
            raise AttributeError(msg)

        
        S_AP = []
        # Initialization:  search the 1st source location over the entire
        # dipoles topographies space
        ap_val1 = np.zeros((n_orders, n_dipoles))
        adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0).toarray()
        L = leadfields[0]
        cluster_indices = []
        # initial_candidate
        norm_1 = np.einsum('ij,jk,ki->i', L.T, C, L)
        norm_2 = np.diag(L.T @ L) # not necessary since leadfields were L2-normalized before
        result = norm_1 / norm_2
        
        last_power = np.max( result )
        cluster_indices.append(np.argmax( result ))
        cluster_indices = np.array(cluster_indices)
        print(f"First index is {cluster_indices[0]} ({cluster_indices})")
        # print(f"adjacency={adjacency}")
        while True:
            new_pot_neighbors = list(set(np.concatenate([np.where(adjacency[ci]==1)[0] for ci in cluster_indices])))
            new_pot_neighbors = np.setdiff1d(new_pot_neighbors, cluster_indices)
            traces = []
            for neighbor in new_pot_neighbors:
                # composite leafield of cluster_indices and neighbor
                L = np.hstack([leadfields[0][:, cluster_indices], leadfields[0][:, neighbor:neighbor+1]])
                U, _, _ = np.linalg.svd(L, full_matrices=False)
                U = U[:, :max_rank]
                norm_1 = np.einsum('ij,jk,ki->i', U.T, C, U)
                norm_2 = np.diag(U.T @ U) # not necessary since leadfields were L2-normalized before
                result = norm_1 / norm_2
                tr_current = np.sum( abs(result) )
                traces.append(tr_current)
            
            if np.max(traces) >= last_power * min_change:
                cluster_indices = np.append( cluster_indices, new_pot_neighbors[np.argmax(traces)] )   
                last_power = np.sum( abs(result) )
                print(f"Added new dipole at {cluster_indices[-1]} because power was higher by {100*np.max(traces)/last_power:.2f}  than {last_power}")
            else:
                print("Stopping")
                break
        
        S_AP.append( cluster_indices )
        # print("S_AP: ", np.concatenate(S_AP))
        for ii in range(1, n_comp):
            

            A = np.stack([leadfields[0][:, dipole] for dipole in np.concatenate(S_AP)], axis=1)
            P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
            Q = np.identity(P_A.shape[0]) - P_A
            QC = Q @ C
            
            result = np.sum(U * (QC @ (Q @ U)), axis=0) / np.sum(U * (Q @ U), axis=0)  # fast and stable
            last_power = np.max( result )
            cluster_indices = np.array([ np.argmax( result ) ])
            print(f"First index of source {ii} is {cluster_indices[0]} ({cluster_indices})")
            
            while True:
                new_pot_neighbors = list(set(np.concatenate([np.where(adjacency[ci]==1)[0] for ci in cluster_indices])))
                new_pot_neighbors = np.setdiff1d(new_pot_neighbors, cluster_indices)
                
                traces = []
                for neighbor in new_pot_neighbors:
                    # composite leafield of cluster_indices and neighbor
                    L = np.hstack([leadfields[0][:, cluster_indices], leadfields[0][:, neighbor:neighbor+1]])
                    U, _, _ = np.linalg.svd(L, full_matrices=False)
                    U = U[:, :max_rank]

                    A = np.stack([leadfields[0][:, dipole] for dipole in np.concatenate(S_AP)], axis=1)
                    P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
                    Q = np.identity(P_A.shape[0]) - P_A
                    QC = Q @ C
                    
                    result = np.sum(U * (QC @ (Q @ U)), axis=0) / np.sum(U * (Q @ U), axis=0)  # fast and stable
                    
                    tr_current = np.sum( abs(result) )
                    traces.append(tr_current)
                
                if np.max(traces) >= last_power * min_change:
                    cluster_indices = np.append( cluster_indices, new_pot_neighbors[np.argmax(traces)] )   
                    last_power = np.sum( abs(result) )
                    print(f"Added new dipole at {cluster_indices[-1]} because power was higher by {100*np.max(traces)/last_power:.2f}  than {last_power}")
                else:
                    print("Stopping")
                    break
        
            S_AP.append( cluster_indices )
        
        # refinement missing
        
        inverse_operator = np.zeros((n_dipoles, n_chans))
        
        nonzero = np.concatenate(S_AP)
        

        L = self.leadfield[:, nonzero]
        if depth_weights is not None:
            L = L * depth_weights[nonzero]
        
        # # Version 2: MNE vanilla
        inverse_operator[nonzero, :] = np.linalg.inv(L.T@L) @ L.T
        
        return inverse_operator

    # def prepare_flex(self):
    #     ''' Create the dictionary of increasingly smooth sources unless self.n_orders==0.
        
    #     Parameters
    #     ----------

    #     '''
    #     n_dipoles = self.leadfield.shape[1]
    #     I = np.identity(n_dipoles)
    #     if self.adjacency_type == "spatial":
    #         adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
    #     else:
    #         adjacency = mne.spatial_dist_adjacency(self.forward['src'], self.adjacency_distance, verbose=None)
        
    #     LL = laplacian(adjacency)
    #     self.leadfields = [deepcopy(self.leadfield), ]
    #     self.gradients = [csr_matrix(I),]
        

    #     if self.diffusion_smoothing:
    #         smoothing_operator = csr_matrix(I - self.diffusion_parameter * LL)
    #     else:
    #         smoothing_operator = csr_matrix(abs(LL))

    #     for _ in range(self.n_orders):
    #         new_leadfield = self.leadfields[-1] @ smoothing_operator
    #         new_gradient = self.gradients[-1] @ smoothing_operator

    #         # Scaling? Not sure...
    #         if self.scale_leadfield:
    #             new_leadfield -= new_leadfield.mean(axis=0)
    #             new_leadfield /= np.linalg.norm(new_leadfield, axis=0)
        
    #         self.leadfields.append( new_leadfield )
    #         self.gradients.append( new_gradient )
    #     # scale and transform gradients
    #     for i in range(self.n_orders+1):
    #         # self.gradients[i] = self.gradients[i].toarray() / self.gradients[i].toarray().max(axis=0)
    #         row_max = self.gradients[i].max(axis=1).toarray().ravel()
    #         scaling_factors = 1 / row_max
    #         self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))
        
    #     self.is_prepared = True

    def prepare_flex(self):
        ''' Create the dictionary of increasingly smooth sources unless
        self.n_orders==0. Flexibly selects diffusion parameter, too.
        
        Parameters
        ----------

        '''
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)
        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(self.forward['src'], self.adjacency_distance, verbose=None)
        
        LL = laplacian(adjacency)
        self.leadfields = [deepcopy(self.leadfield), ]
        self.gradients = [csr_matrix(I),]
        

        # if self.diffusion_smoothing:
        #     smoothing_operator = csr_matrix(I - self.diffusion_parameter * LL)
        # else:
        #     smoothing_operator = csr_matrix(abs(LL))
        if self.diffusion_parameter == "auto":
            alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            smoothing_operators = [csr_matrix(I - alpha * LL) for alpha in alphas]
        else:
            smoothing_operators = [csr_matrix(I - self.diffusion_parameter * LL),]


        for smoothing_operator in smoothing_operators:

            for i in range(self.n_orders):
                smoothing_operator_i = smoothing_operator**(i+1)  # csr_matrix(np.linalg.matrix_power(smoothing_operator.toarray(), i+1))
                new_leadfield = self.leadfields[0] @ smoothing_operator_i
                new_gradient = self.gradients[0] @ smoothing_operator_i

                # Scaling? Not sure...
                if self.scale_leadfield:
                    # new_leadfield -= new_leadfield.mean(axis=0)
                    new_leadfield /= np.linalg.norm(new_leadfield, axis=0)
            
                self.leadfields.append( new_leadfield )
                self.gradients.append( new_gradient )
        # scale and transform gradients
        for i in range(len(self.gradients)):
            # self.gradients[i] = self.gradients[i].toarray() / self.gradients[i].toarray().max(axis=0)
            # row_max = self.gradients[i].max(axis=1).toarray().ravel()
            # scaling_factors = 1 / row_max
            row_sums = self.gradients[i].sum(axis=1).ravel()  # Compute the sum of each row
            scaling_factors = 1 / row_sums
            self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))
        
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

    def make_inverse_operator(self, forward, mne_obj, *args, alpha="auto", n="auto", 
                              k="auto", stop_crit=0.95, distance_weighting=False, **kwargs):
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
        
        inverse_operator = self.make_flex(data, n, k, stop_crit, self.truncate, distance_weighting=distance_weighting)
        
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]
        return self

    def make_flex(self, y, n, k, stop_crit, truncate, distance_weighting=False):
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
        # leadfield -= leadfield.mean(axis=0)

        if k == "auto":
            k = n_chans
        # Assert common average reference
        # y -= y.mean(axis=0)
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
            Ps = Us@Us.T
            selected_idc = []

            norm_1 = np.linalg.norm(Ps @ Q @ leadfield, axis=0)
            norm_2 = np.linalg.norm(Q @ leadfield, axis=0) 
            mu = norm_1 / norm_2
            selected_idc.append( np.argmax(mu) )
            current_max = np.max(mu)
            current_leadfield = leadfield[:, selected_idc[0]]
            component_strength = [1,]

            while True:
                # Find all neighboring dipoles of current candidates
                neighbors = np.unique(np.concatenate([self.members_ex[idx] for idx in selected_idc]))
                # Filter out candidates from neighbors
                for idx, n in reversed(list(enumerate(neighbors))):
                    if n in selected_idc:
                        neighbors = np.delete(neighbors, idx)

                dist = self.distances[neighbors, selected_idc[0]]

                # construct new candidate leadfields:
                if distance_weighting:
                    b = np.stack([ leadfield[:, n] / d + current_leadfield for d, n in zip(dist, neighbors)], axis=1)
                else:
                    b = np.stack([ leadfield[:, n] + current_leadfield for d, n in zip(dist, neighbors)], axis=1)
                    
                # b /= np.linalg.norm(b, axis=0)
                # norm_1 = np.linalg.norm(Ps @ Q @ b, axis=0)
                # norm_2 = np.linalg.norm(Q @ b, axis=0) 
                norm_1 = np.linalg.norm(abs(Ps) @ abs(Q) @ abs(b), axis=0)
                norm_2 = np.linalg.norm(abs(Q) @ abs(b), axis=0) 
                
                mu_b = norm_1 / norm_2
                max_mu_b = np.max(mu_b)
                
                if (max_mu_b - current_max) < 0.00:
                # if max_mu_b / current_max < 1.0001:
                    break
                else:
                    new_idx = neighbors[np.argmax(mu_b)]
                    b_best = b[:, np.argmax(mu_b)]
                    current_max = max_mu_b
                    current_leadfield = b_best #/ np.linalg.norm(b_best)
                    if distance_weighting:
                        component_strength.append( 1/dist[np.argmax(mu_b)] )
                    else:
                        component_strength.append( 1 )
                    
                    selected_idc.append( new_idx )
                            
            # Find the dipole/ patch with highest correlation with the residual
            
            if i>0 and current_max < stop_crit:
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
        # source_covariance = csr_matrix(np.diag(source_covariance))
        # L_s = self.leadfield @ source_covariance
        # L = self.leadfield
        # W = np.diag(np.linalg.norm(L, axis=0)) 
        # # print(source_covariance.shape, L.shape, W.shape)
        # inverse_operator = source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T
        
        Gamma = csr_matrix(np.diag(source_covariance))
        Gamma_LT = Gamma @ leadfield.T
        Sigma_y = leadfield @ Gamma_LT
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma_LT @ Sigma_y_inv

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


class SolverGeneralizedIterative(BaseSolver):
    ''' Class that generalizes iterative solutions like RAP-MUSIC, AP and SSM
        inverse solution with flexible extent estimation (FLEX-AP).
    
    Attributes
    ----------
    n_orders : int
        Controls the maximum smoothness to pursue.
    
    References
    ---------
    [1] Wax, M., & Adler, A. (2021). Direction of arrival estimation in the
    presence of model errors by signal subspace matching. Signal Processing,
    181, 107900. [2] TBD.

    '''
    def __init__(self, name="Flexible Signal Subspace Matching", scale_leadfield=False, **kwargs):
        self.name = name
        self.is_prepared = False
        self.scale_leadfield = scale_leadfield
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, inverse_type="SSM", n_orders=3,
                              alpha="auto", n="auto", k="auto", refine_solution=True, 
                              max_iter=5, diffusion_smoothing=True, 
                              diffusion_parameter=0.1, adjacency_type="spatial", 
                              adjacency_distance=3e-3, lambda_reg1=0.001, 
                              lambda_reg2=0.0001, lambda_reg3=0.0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        inverse_type : str
            The type of inverse solution to use. "SSM" -> Signal Subspace Matching.
            "AP" -> Alternating Projections. "RAP" -> Recursively Applied and Projected MUSIC.
        n_orders : int
            Controls the maximum smoothness to pursue.
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
        max_iter : int
            Maximum number of iterations during refinement.
        diffusion_smoothing : bool
            Whether to use diffusion smoothing. Default is True.
        diffusion_parameter : float
            The diffusion parameter (alpha). Default is 0.1.
        adjacency_type : str
            The type of adjacency. "spatial" -> based on graph neighbors. "distance" -> based on distance
        adjacency_distance : float
            The distance at which neighboring dipoles are considered neighbors.
        depth_weights : numpy.ndarray
            The depth weights to use for depth weighting the leadfields. If None, no depth weighting is applied.

        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)

        self.diffusion_smoothing = diffusion_smoothing, 
        self.diffusion_parameter = diffusion_parameter
        self.n_orders = n_orders
        self.adjacency_type = adjacency_type
        self.adjacency_distance = adjacency_distance
        self.inverse_type = inverse_type
        data = self.unpack_data_obj(mne_obj)

        if not self.is_prepared:
            self.prepare_flex()
        
        if inverse_type == "SSM":
            self.get_source = self.get_source_ssm
            self.get_covariance = self.get_covariance_ssm
        elif inverse_type == "AP":
            self.get_source = self.get_source_ap
            self.get_covariance = self.get_covariance_ap
        elif inverse_type == "RAP":
            self.get_source = self.get_source_rap
            self.get_covariance = self.get_covariance_rap
        else:
            self.get_source = None
            raise AttributeError(f"Unknown inverse_type: {inverse_type}")

        inverse_operator = self.make_iterative_solution(data, n, k, 
                                        max_iter=max_iter, 
                                        refine_solution=refine_solution,
                                        lambda_reg1=lambda_reg1,
                                        lambda_reg2=lambda_reg2,
                                        lambda_reg3=lambda_reg3)
        self.inverse_operators = [InverseOperator(inverse_operator, self.name), ]

        return self

    def make_iterative_solution(self, Y, n, k, refine_solution=True, max_iter=5, 
                 lambda_reg1=0.001, lambda_reg2=0.0001, lambda_reg3=0.0):
        ''' Create the iterative inverse solution to the EEG data.
        
        Parameters
        ----------
        Y : numpy.ndarray
            EEG data matrix (channels, time)
        n : int/ str
            Number of eigenvectors to use or "auto" for l-curve method.
        k : int
            Number of recursions.
        refine_solution : bool
            If True: Re-visit each selected candidate and check if there is a
            better alternative.
        lambda_reg1 : float
            Regularization parameter for the projection matrix.
        lambda_reg2 : float
            Regularization parameter for the source covariance matrix.
        lambda_reg3 : float
            Regularization parameter for the source covariance matrix.

        Return
        ------
        x_hat : numpy.ndarray
            Source data matrix (sources, time)
        '''
        n_chans, n_dipoles = self.leadfield.shape
        n_time = Y.shape[1]

        leadfields = self.leadfields
        n_orders = len(self.leadfields)

        if k == "auto":
            k = n_chans
        
        # Determine the number of sources
        if n == "auto":
            n_comp = self.estimate_comps(Y)
        else:
            n_comp = deepcopy(n)
        
        P_Y = self.get_covariance(Y, n_comp, lambda_reg1=lambda_reg1)
        Q = np.eye(n_chans)  # np.zeros((n_chans, n_chans))

        candidates = []
        A_q = []
        
        # Initial source location
        expression = self.get_source(P_Y, Q, leadfields, lambda_reg=lambda_reg3)
        order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
        candidates.append( [order, dipole] )
        print(order, dipole)

        # Now, add one source at a time
        for qq in range(1, n_comp):
            order, location = candidates[-1]
            A_q.append(leadfields[order][:, location])
            Q = self.compute_projection_matrix(A_q, lambda_reg=lambda_reg2)
            expression = self.get_source(P_Y, Q, leadfields, candidates, lambda_reg=lambda_reg3)
            order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
            candidates.append( [order, dipole] )
        
        A_q.append( leadfields[order][:, dipole] )
        
        # Phase 2: refinement
        candidates_2 = deepcopy(candidates)
        if len(candidates_2) > 1 and refine_solution:
            candidates_2_prev = deepcopy(candidates_2)
            for j in range(max_iter):
                A_q_j = A_q.copy()
                for qq in range(n_comp):
                    A_temp = np.delete(A_q_j, qq, axis=0) # delete the current source
                    qq_temp = np.delete(candidates_2, qq, axis=0) # delete the current source
                    Q = self.compute_projection_matrix(A_temp, lambda_reg=lambda_reg2)
                    expression = self.get_source(P_Y, Q, leadfields, qq_temp, lambda_reg=lambda_reg3)
                    order, dipole = np.unravel_index(np.nanargmax(expression), expression.shape)
                    candidates_2[qq] = [order, dipole]
                    A_q_j[qq] = leadfields[candidates_2[qq][0]][:, candidates_2[qq][1]]

                if candidates_2 == candidates_2_prev:
                    # print(f"No change after {j+1} iterations")
                    break
                # else:
                #     print(candidates_2_prev, " ==> ", candidates_2)
                candidates_2_prev = deepcopy(candidates_2)

        self.candidates = candidates_2       

        # Low-rank minimum norm solution
        source_covariance = np.identity(n_comp)
        L = np.stack([leadfields[order][:, dipole] for order, dipole in candidates_2], axis=1)
        gradients = np.squeeze(np.stack([self.gradients[order][dipole].toarray() for order, dipole in candidates_2], axis=1))
        inverse_operator = gradients.T @ source_covariance @ L.T @ np.linalg.pinv(L @ source_covariance @ L.T)
        
        return inverse_operator
   
    def estimate_comps(self, Y, n):
        ''' Estimate the number of sources to use.
        Parameters
        ----------
        Y : numpy.ndarray
            The data matrix.
        n : str
            The method to use. "L" -> L-curve method. "drop" -> based on eigenvalue drop-off.
            "auto" -> Combine L and drop method. "mean" -> Selects the eigenvalues that are larger than the mean of all eigs.
        Return
        ------
        n_comp : int
            The number of sources to use.
        
        '''
        C = Y @ Y.T
        _, D, _= np.linalg.svd(C, full_matrices=False)
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
            n_comp_mean = np.where(D<D.mean())[0][0]

            # Combine the two and bias it slightly:
            n_comp = np.ceil(1.1*np.mean([n_comp_L, n_comp_drop, n_comp_mean])).astype(int)
        else:
            raise AttributeError(f"Unknown n: {n}")
        return n_comp
    
    @staticmethod
    def get_covariance_ssm(Y, *args, lambda_reg1=0.001, **kwargs):
        ''' Compute the source covariance matrix for the SSM algorithm.'''
        n_time = Y.shape[1]

        M_Y = Y.T @ Y
        YY = M_Y + lambda_reg1 * np.trace(M_Y) * np.eye(n_time)
        P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
        return P_Y
    
    @staticmethod
    def get_covariance_rap(Y, n_comp, *args, lambda_reg1=0.001, **kwargs):
        ''' Compute the source covariance matrix for the SSM algorithm.'''
        n_chans, n_time = Y.shape[1]
        
        C = Y @ Y.T
        U, _, _= np.linalg.svd(C, full_matrices=False)
        Us = U[:, :n_comp]
        
        # MUSIC subspace-based Covariance
        P_Y = Us @ Us.T

        return P_Y

    @staticmethod
    def get_covariance_ap(Y, *args, lambda_reg1=0.001, **kwargs):
        ''' Compute the source covariance matrix for the SSM algorithm.'''
        n_chans, n_time = Y.shape
        C = Y @ Y.T
        P_Y = C + lambda_reg1 * np.trace(C) * np.eye(n_chans)
        return P_Y
    
    @staticmethod
    def get_source_ssm(P_Y: np.ndarray, Q: np.ndarray, leadfields: list, q_ignore: list=[], lambda_reg=0.0):
        ''' Compute the source with the highest AP value.
        Parameters
        ----------
        P_Y : numpy.ndarray
            The projection matrix of the data covariance.
        Q : numpy.ndarray
            The out-projection matrix of the source covariance.
        leadfields : list
            The list of leadfields.
        q_ignore : list
            List of sources to ignore (e.g., already selected sources).
        lambda_reg : float
            Regularization parameter to enhance stability.
        '''
        # print(f"Check within function")
        # print(P_Y.shape, P_Y)
        # print(Q.shape, Q)
        # print(leadfields[0].shape, leadfields[0])
        # print(q_ignore, lambda_reg)

        n_dipoles = leadfields[0].shape[1]
        n_orders = len(leadfields)

        P_Y_T_P_Y = P_Y.T @ P_Y
        
        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            a_s = Q @ leadfields[jj]

            # Fast
            upper = np.einsum('ij,ij->j', a_s, P_Y_T_P_Y @ a_s)
            lower = np.einsum('ij,ij->j', a_s, a_s) + lambda_reg
            expression[jj, :] = upper  / lower

        if q_ignore != []:
            for order, dipole in q_ignore:
                expression[order, dipole] = np.nan

        return expression
    
    @staticmethod
    def get_source_rap(P_Y: np.ndarray, Q: np.ndarray, leadfields: list, q_ignore: list=[], lambda_reg=0.0):
        n_orders = len(leadfields)
        n_chans, n_dipoles = leadfields[0].shape
        P_YQ = P_Y @ Q
        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            L = leadfields[jj]
            norm_1 = np.linalg.norm(P_YQ @ L, axis=0)
            norm_2 = np.linalg.norm(Q @ L, axis=0)
            expression[jj, :] = norm_1 / norm_2

        return expression
    
    @staticmethod
    def get_source_ap(C, Q, leadfields, q_ignore: list=[], **kwargs):
        ''' Compute the source with the highest AP value.
        Parameters
        ----------
        C : numpy.ndarray
            The data covariance matrix.
        Q : numpy.ndarray
            The out-projection matrix of the source covariance.
        leadfields : list
            The list of leadfields.
        q_ignore : list
            List of sources to ignore (e.g., already selected sources).
        
        Returns
        -------
        order : int
            The order of the selected source.
        dipole : int
            The dipole index of the selected source.

        '''
        # print(Q)
        # print()
        # print(C)

        n_dipoles = leadfields[0].shape[1]
        n_orders = len(leadfields)
        QC = Q @ C
        
        expression = np.zeros((n_orders, n_dipoles))
        for jj in range(n_orders):
            L = leadfields[jj]
            expression[jj, :] = np.sum(L * (QC @ (Q @ L)), axis=0) / np.sum(L * (Q @ L), axis=0)  # fast and stable
            
        if q_ignore != []:
            for order, dipole in q_ignore:
                expression[order, dipole] = np.nan

        return expression

    @staticmethod
    def compute_projection_matrix(A_q, lambda_reg=0.0001):
        ''' Compute the out-projection matrix Q.'''

        A_q = np.stack(A_q, axis=1)
        M_A = A_q.T @ A_q
        AA = M_A + lambda_reg * np.trace(M_A) * np.eye(M_A.shape[0])
        P_A = (A_q @ np.linalg.pinv(AA)) @ A_q.T
        Q = np.eye(P_A.shape[0]) - P_A
        return Q

    def prepare_flex(self):
        ''' Create the dictionary of increasingly smooth sources unless
        self.n_orders==0. Flexibly selects diffusion parameter, too.
        
        Parameters
        ----------

        '''
        n_dipoles = self.leadfield.shape[1]
        I = np.identity(n_dipoles)
        if self.adjacency_type == "spatial":
            adjacency = mne.spatial_src_adjacency(self.forward['src'], verbose=0)
        else:
            adjacency = mne.spatial_dist_adjacency(self.forward['src'], self.adjacency_distance, verbose=None)
        
        LL = laplacian(adjacency)
        self.leadfields = [deepcopy(self.leadfield), ]
        self.gradients = [csr_matrix(I),]

        if self.diffusion_parameter == "auto":
            alphas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            smoothing_operators = [csr_matrix(I - alpha * LL) for alpha in alphas]
        else:
            smoothing_operators = [csr_matrix(I - self.diffusion_parameter * LL),]


        for smoothing_operator in smoothing_operators:

            for i in range(self.n_orders):
                smoothing_operator_i = smoothing_operator**(i+1)  # csr_matrix(np.linalg.matrix_power(smoothing_operator.toarray(), i+1))
                new_leadfield = self.leadfields[0] @ smoothing_operator_i
                new_gradient = self.gradients[0] @ smoothing_operator_i

                # Scaling? Not sure...
                if self.scale_leadfield:
                    # new_leadfield -= new_leadfield.mean(axis=0)
                    new_leadfield /= np.linalg.norm(new_leadfield, axis=0)
            
                self.leadfields.append( new_leadfield )
                self.gradients.append( new_gradient )
        # scale and transform gradients
        for i in range(len(self.gradients)):
            row_sums = self.gradients[i].sum(axis=1).ravel()  # Compute the sum of each row
            scaling_factors = 1 / row_sums
            self.gradients[i] = csr_matrix(self.gradients[i].multiply(scaling_factors.reshape(-1, 1)))
        
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
