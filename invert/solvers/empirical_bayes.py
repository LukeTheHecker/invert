from copy import deepcopy
from scipy.spatial.distance import cdist
from scipy.sparse import spdiags
from scipy.linalg import inv
import numpy as np
import mne
from scipy.fftpack import dct
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from ..util import pos_from_forward
# from ..invert import BaseSolver, InverseOperator
# from .. import invert
from .base import BaseSolver, InverseOperator

# from .. import invert
# import BaseSolver, InverseOperator


class SolverChampagne(BaseSolver):
    ''' Class for the Champagne inverse solution. Code is based on the
    implementation from the BSI-Zoo: https://github.com/braindatalab/BSI-Zoo/
    
    References
    ----------
    [1] Owen, J., Attias, H., Sekihara, K., Nagarajan, S., & Wipf, D. (2008).
    Estimating the location and orientation of complex, correlated neural
    activity using MEG. Advances in Neural Information Processing Systems, 21.
    
    [2] Wipf, D. P., Owen, J. P., Attias, H. T., Sekihara, K., & Nagarajan, S.
    S. (2010). Robust Bayesian estimation of the location, orientation, and time
    course of multiple correlated neural sources using MEG. NeuroImage, 49(1),
    641-655. 
    
    [3] Owen, J. P., Wipf, D. P., Attias, H. T., Sekihara, K., &
    Nagarajan, S. S. (2012). Performance evaluation of the Champagne source
    reconstruction algorithm on simulated and real M/EEG data. Neuroimage,
    60(1), 305-323.
    '''

    def __init__(self, name="Champagne", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', max_iter=1000, noise_cov=None, verbose=0, **kwargs):
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
        self.leadfield = self.forward['sol']['data']
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.champagne(evoked.data, alpha, max_iter=max_iter,)
            inverse_operators.append( inverse_operator )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:

        return super().apply_inverse_operator(evoked)
    
    
    def champagne(self, y, alpha, max_iter=1000):
        """Champagne method based on our MATLAB codes  
        -> copied as mentioned in class docstring

        Parameters
        ----------
        y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        max_iter : int, optional
            The maximum number of inner loop iterations

        Returns
        -------
        x : array, shape (n_sources,)
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        
        """
        _, n_sources = self.leadfield.shape
        _, n_times = y.shape
        leadfield = deepcopy(self.leadfield)
        gammas = np.ones(n_sources)
        eps = np.finfo(float).eps
        threshold = 0.2 * np.mean(np.diag(self.noise_cov))
        # x = np.zeros((n_sources, n_times))
        n_active = n_sources
        active_set = np.arange(n_sources)
        # H = np.concatenate(L, np.eyes(n_sensors), axis = 1)
        self.noise_cov = alpha*self.noise_cov
        x_bars = []
        for i in range(max_iter):
            gammas[np.isnan(gammas)] = 0.0
            gidx = np.abs(gammas) > threshold
            active_set = active_set[gidx]
            gammas = gammas[gidx]

            # update only active gammas (once set to zero it stays at zero)
            if n_active > len(active_set):
                n_active = active_set.size
                leadfield = leadfield[:, gidx]

            Gamma = spdiags(gammas, 0, len(active_set), len(active_set))
            # Calculate Source Covariance Matrix based on currently selected gammas
            Sigma_y = (leadfield @ Gamma @ leadfield.T) + self.noise_cov
            U, S, _ = np.linalg.svd(Sigma_y, full_matrices=False)
            S = S[np.newaxis, :]
            del Sigma_y
            Sigma_y_inv = np.dot(U / (S + eps), U.T)
            # Sigma_y_inv = linalg.inv(Sigma_y)
            x_bar = Gamma @ leadfield.T @ Sigma_y_inv @ y

            # old gamma calculation throws warning
            # gammas = np.sqrt(
            #     np.diag(x_bar @ x_bar.T / n_times) / np.diag(leadfield.T @ Sigma_y_inv @ leadfield)
            # )
            # Calculate gammas 
            gammas = np.diag(x_bar @ x_bar.T / n_times) / np.diag(leadfield.T @ Sigma_y_inv @ leadfield)
            # set negative gammas to nan to avoid bad sqrt
            gammas.astype(np.float64)  # this is required for numpy to accept nan
            gammas[gammas<0] = np.nan
            gammas = np.sqrt(gammas)

            # Calculate Residual to the data
            e_bar = y - (leadfield @ x_bar)
            self.noise_cov = np.sqrt(np.diag(e_bar @ e_bar.T / n_times) / np.diag(Sigma_y_inv))
            threshold = 0.2 * np.mean(np.diag(self.noise_cov))
            x_bars.append(x_bar)

            if i>0 and np.linalg.norm(x_bars[-1]) == 0:
                x_bar = x_bars[-2]
                break
        active_set
        gammas_full = np.zeros(n_sources)
        gammas_full[active_set] = gammas
        Gamma_full = spdiags(gammas_full, 0, n_sources, n_sources)
        Sigma_y = (self.leadfield @ Gamma_full @ self.leadfield.T) + self.noise_cov
        U, S, _ = np.linalg.svd(Sigma_y, full_matrices=False)
        S = S[np.newaxis, :]
        del Sigma_y
        Sigma_y_inv_full = np.dot(U / (S + eps), U.T)
        inverse_operator = Gamma_full @ self.leadfield.T @ Sigma_y_inv_full

        return inverse_operator

# class SolverChampagne(BaseSolver):
#     ''' Class for the Champagne inverse solution. Code is based on the
#     implementation from the BSI-Zoo: https://github.com/braindatalab/BSI-Zoo/
    
#     References
#     ----------
#     [1] Owen, J., Attias, H., Sekihara, K., Nagarajan, S., & Wipf, D. (2008).
#     Estimating the location and orientation of complex, correlated neural
#     activity using MEG. Advances in Neural Information Processing Systems, 21.
    
#     [2] Wipf, D. P., Owen, J. P., Attias, H. T., Sekihara, K., & Nagarajan, S.
#     S. (2010). Robust Bayesian estimation of the location, orientation, and time
#     course of multiple correlated neural sources using MEG. NeuroImage, 49(1),
#     641-655. 
    
#     [3] Owen, J. P., Wipf, D. P., Attias, H. T., Sekihara, K., &
#     Nagarajan, S. S. (2012). Performance evaluation of the Champagne source
#     reconstruction algorithm on simulated and real M/EEG data. Neuroimage,
#     60(1), 305-323.
#     '''

#     def __init__(self, name="Champagne", **kwargs):
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
#         self.forward = forward
#         self.leadfield = self.forward['sol']['data']
#         super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
#         n_chans = self.leadfield.shape[0]
#         if noise_cov is None:
#             noise_cov = np.identity(n_chans)

#         self.noise_cov = noise_cov
#         self.inverse_operators = []
#         return self

#     def apply_inverse_operator(self, evoked, max_iter=1000) -> mne.SourceEstimate:

#         source_mat = self.champagne(evoked.data, max_iter=max_iter)
#         stc = self.source_to_object(source_mat, evoked)
#         return stc
    
    
#     def champagne(self, y, max_iter=1000):
#         """Champagne method based on our MATLAB codes  
#         -> copied as mentioned in class docstring

#         Parameters
#         ----------
#         y : array, shape (n_sensors,)
#             measurement vector, capturing sensor measurements
#         max_iter : int, optional
#             The maximum number of inner loop iterations

#         Returns
#         -------
#         x : array, shape (n_sources,)
#             Parameter vector, e.g., source vector in the context of BSI (x in the cost
#             function formula).
        
#         """
#         _, n_sources = self.leadfield.shape
#         _, n_times = y.shape
#         if self.alpha == "auto":
#             self.alpha = 1
#         gammas = np.ones(n_sources)
#         eps = np.finfo(float).eps
#         threshold = 0.2 * np.mean(np.diag(self.noise_cov))
#         x = np.zeros((n_sources, n_times))
#         n_active = n_sources
#         active_set = np.arange(n_sources)
#         # H = np.concatenate(L, np.eyes(n_sensors), axis = 1)
#         self.noise_cov = self.alpha*self.noise_cov
#         x_bars = []
#         for i in range(max_iter):
#             gammas[np.isnan(gammas)] = 0.0
#             gidx = np.abs(gammas) > threshold
#             active_set = active_set[gidx]
#             gammas = gammas[gidx]

#             # update only active gammas (once set to zero it stays at zero)
#             if n_active > len(active_set):
#                 n_active = active_set.size
#                 self.leadfield = self.leadfield[:, gidx]

#             Gamma = spdiags(gammas, 0, len(active_set), len(active_set))
#             # Calculate Source Covariance Matrix based on currently selected gammas
#             Sigma_y = (self.leadfield @ Gamma @ self.leadfield.T) + self.noise_cov
#             U, S, _ = np.linalg.svd(Sigma_y, full_matrices=False)
#             S = S[np.newaxis, :]
#             del Sigma_y
#             Sigma_y_inv = np.dot(U / (S + eps), U.T)
#             # Sigma_y_inv = linalg.inv(Sigma_y)
#             x_bar = Gamma @ self.leadfield.T @ Sigma_y_inv @ y

#             gammas = np.sqrt(
#                 np.diag(x_bar @ x_bar.T / n_times) / np.diag(self.leadfield.T @ Sigma_y_inv @ self.leadfield)
#             )
#             # Calculate Residual to the data
#             e_bar = y - (self.leadfield @ x_bar)
#             self.noise_cov = np.sqrt(np.diag(e_bar @ e_bar.T / n_times) / np.diag(Sigma_y_inv))
#             threshold = 0.2 * np.mean(np.diag(self.noise_cov))
#             x_bars.append(x_bar)

#             if i>0 and np.linalg.norm(x_bars[-1]) == 0:
#                 x_bar = x_bars[-2]
#                 break

#         x[active_set, :] = x_bar

#         return x

class SolverMultipleSparsePriors(BaseSolver):
    ''' Class for the Multiple Sparse Priors (MSP) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Multiple Sparse Priors", inversion_type="MSP", **kwargs):
        if inversion_type == "MSP":
            self.name = name
        elif inversion_type == "LORETA":
            self.name = "Bayesian LORETA"
        elif inversion_type == "MNE":
            self.name = "Bayesian Minimum Norm Estimates"
        elif inversion_type == "BMF":
            self.name = "Bayesian Beamformer"
        elif inversion_type == "BMF-LOR":
            self.name = "Bayesian Beamformer + LORETA"
        self.inversion_type = inversion_type
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, Np=64, 
                              max_iter=128, smoothness=0.6, alpha='auto', 
                              verbose=0, **kwargs):
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
        leadfield = self.forward['sol']['data']
        pos = pos_from_forward(forward, verbose=verbose)
        adjacency = mne.spatial_src_adjacency(forward['src'], verbose=verbose).toarray()
        Y = evoked.data
        A = get_spatial_projector(leadfield)
        
        S, V = get_temporal_projector(evoked, leadfield, A)
                
        Y_ = A @ Y @ S
        
        

        leadfield_ = A @ leadfield
        maximum_a_posteriori = make_msp_map(Y_, leadfield_, pos, adjacency, A, Np=Np, max_iter=max_iter, 
            inversion_type=self.inversion_type, smoothness=smoothness)
        inverse_operators = [maximum_a_posteriori, A, S]
        
        
        
        self.inverse_operators = [InverseOperator(inverse_operators, self.name),]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        S = self.inverse_operators[0].data[-1]
        if S.shape[1] != evoked.data.shape[1]:
            # print("\tRe-calculating projectors...")
            self.make_inverse_operator(self.forward, evoked)
            # print("\T...done!")
        return super().apply_inverse_operator(evoked)

def make_msp_inverse_operator(leadfield, pos, adjacency, evoked,Np=64, 
                              max_iter=128, inversion_type='MSP', 
                              smoothness=0.6, noise_cov=None, **kwargs):
    """ Calculate the inverse operator using Multiple Sparse Priors.

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    alpha : float
        The regularization parameter.
    noise_cov : numpy.ndarray
        The noise covariance matrix (channels x channels).

    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    n_chans, _ = leadfield.shape
    if noise_cov is None:
        noise_cov = np.identity(n_chans)

    A = get_spatial_projector(leadfield)
    S, V = get_temporal_projector(evoked, leadfield, A)
    Y = evoked.data

    Y_ = A @ Y @ S
    leadfield_ = A @ leadfield
    maximum_a_posteriori = make_msp_map(Y_, leadfield_, pos, adjacency, A, Np=Np, max_iter=max_iter, 
        inversion_type=inversion_type, smoothness=smoothness)
    inverse_operator = [maximum_a_posteriori, A, S]
    # J_ = maximum_a_posteriori @ Y_
    # D_MSP =  J_ @ S.T 
    return inverse_operator


def inverse_msp(evoked, fwd, Np=64, max_iter=128, inversion_type='MSP', smoothness=0.6):
    leadfield, pos = unpack_fwd(fwd)[1:3]
    A = get_spatial_projector(leadfield)
    S, V = get_temporal_projector(evoked, leadfield, A)
    Y = evoked.data

    Y_ = A @ Y @ S
    leadfield_ = A @ leadfield
    maximum_a_posteriori = make_msp_map(Y_, leadfield_, fwd, A, Np=Np, max_iter=max_iter, 
        inversion_type=inversion_type, smoothness=smoothness)
    J_ = maximum_a_posteriori @ Y_
    D_MSP =  J_ @ S.T 
    return D_MSP



def get_spatial_projector(leadfield):
    # eliminate low SNR spatial modes
    thr = np.e**(-16)
    U,s2,v2    = np.linalg.svd(leadfield @ leadfield.T)
    # This replaces the custom svd function from spm called "spm_svd(...):"
    s2 *=  s2
    keep = (s2*len(s2) / s2.sum()) > thr
    # This replaces the custom svd function from spm called "spm_svd(...):"
    s2 = s2[keep]
    U = U[:, keep]
    v2 = v2[:,keep]
    del s2, v2

    A     = U.T;  # spatial projector A
    # UL    = np.matmul(A,L)
    # Nm    = UL.shape[0]				# Number of spatial projectors
    # Is = np.arange(Nd)
    # Ns = len(Is)
    return A

def get_temporal_projector(evoked, leadfield, A, Nmax=16, hpf=0, lpf=45, sdv=4):
    '''
    Parameters
    ----------
    Nmax : int
        max number of temporal modes
    '''
    Nd = leadfield.shape

    #  Time-window of interest
    It = np.arange(len(evoked.times)).astype(int)
    # print(It)
    # % Peristimulus time
    pst    = 1000*evoked.times  # peristimulus time (ms)
    pst    = pst[It]  # windowed time (ms)
    dur    = (pst[-1] - pst[0])/1000  # duration (s)
    dct    = (It - It[0])/2/dur  # DCT frequencies (Hz)
    Nb     = len(It)  # number of time bins
    # dct
    # # Serial correlations
    K      = np.e**(-(pst - pst[0])**2/(2*sdv**2));
    K      = toeplitz(K);
    qV     = np.matmul(K,K.T)
    # # Confounds and temporal subspace
    T      = spm_dctmtx(Nb,Nb)  # use plot(T) here!
    j      = (dct>=hpf) & (dct <=lpf)  # Filter
    T      = T[:,j]  # Apply the filter to discrete cosines
    dct    = dct[j]  # Frequencies accepted

    # No Hanning window
    # W  = sparse(1:Nb,1:Nb,spm_hanning(Nb));
    W  = 1  # Apply Hanning if desired!

    # get temporal covariance (Y'*Y) to find temporal modes
    # Note: The variable YY was replaced with YTY because it is
    # duplicated in the original script, causing confusion.
    D_tmp = evoked.data
    D_tmp = D_tmp[:, :]
    # print(D_tmp.shape)
    Y = np.matmul(A, D_tmp)  # Data samples in spatial modes (virtual sensors)
    YTY = np.matmul(Y.T,Y)  # Covariance in temporal domain

    # Apply any Hanning and filtering
    # YTY = np.matmul(np.matmul(W, YTY), W)  # Hanning
    YTY = np.matmul(np.matmul(T.T, YTY), T)  # Filter

    # Plot temporal projectors
    # plt.figure()
    # plt.imshow(YTY, extent=[dct.min(), dct.max(), dct.min(), dct.max()])  # Plot of frequency map
    # plt.title('Temporal projector')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Frequency (Hz)')

    # temporal projector (at most Nrmax modes) S = T*V
    # [U E]  = spm_svd(YTY,exp(-8));		# get temporal modes
    U, E, _  = np.linalg.svd(YTY);		# get temporal modes
    thr = np.e**(-16)
    E *=  E
    keep = (E*len(E) / E.sum()) > thr
    # print(keep.sum()/len(keep), E)
    # keep = E>thr
    # U = U[:, keep]
    E = E[keep]

    # This replaces the custom svd function from spm called "spm_svd(...):"

    E      = np.diag(E)/np.trace(YTY)  # normalise variance
    Nr     = np.min([len(E),Nmax]) 	# number of temporal modes
    V      = U[:,:Nr]  # temporal modes
    VE     = np.sum(E[:Nr])  # variance explained

    # print(f'Using {Nr} temporal mode(s)',)
    # print(f'accounting for {100*VE:.1f} % average variance\n')

    # projection and whitening
    # print(YTY.shape, T.shape, Y.shape, V.shape)
    S      = T @ V  # temporal projector
    # Vq     = np.matmul(S, np.matmul(np.linalg.pinv(np.matmul(np.matmul(S.T, qV),S)), S.T))  # temporal precision

    # # stack (scaled aligned data) over modalities
    # Y    = np.matmul(np.matmul(A, D_tmp), S)  # Load data in spatial and temporal modes

    # # accumulate first & second-order responses
    # YY   = np.matmul(Y, Y.T)  # Data covariance in spatial mode

    # # generate sensor error components (Qe)
    # # assuming equal noise over subjects (Qe) and modalities AQ
    # AQeA   = np.matmul(A,A.T)  # Note that here it is A*A', it means
    # Qe  = AQeA/(np.trace(AQeA))  # IID noise in virtual sensor space
    return S, V

def greens_function(adjacency, smoothness=0.6):
    
    A = adjacency
    Nd = A.shape[0]
    GL = A - spdiags( A.sum(axis=1), 0, Nd, Nd)
    GL = GL * smoothness/2


    Qi = np.identity(Nd)
    QG = np.zeros((Nd, Nd))
    for i in range(8):
        QG += Qi
        Qi = Qi @ GL / (i+1)
    
    QG = QG * (QG > np.exp(-8))
    QG = QG @ QG
    return QG

def make_msp_map(Y, leadfield, pos, adjacency, A, Np=128, smoothness=0.6, max_iter=128,
    inversion_type='MSP'):
    '''
    Create the maximum a posteriori (MAP) estimator for the multiple sparse
    priors inverse solution.


    References
    ----------
    [1] Friston, K., Harrison, L., Daunizeau, J., Kiebel, S., Phillips, C.,
    Trujillo-Barreto, N., ... & Mattout, J. (2008). Multiple sparse priors for
    the M/EEG inverse problem. NeuroImage, 39(3), 1104-1120. 
    
    [2] López, J. D., Litvak, V., Espinosa, J. J., Friston, K., & Barnes, G. R.
    (2014). Algorithmic procedures for Bayesian MEG/EEG source reconstruction in
    SPM. NeuroImage, 84, 476-487.
    '''
    n, d = leadfield.shape
    # u, v = Y.shape
    C = Y @ Y.T  # sensor covariance
    # Scale C:
    sY = n*np.trace(C)
    C /= sY

    # v = 1
    # Greens function
    if smoothness is not None:
        Q_G = greens_function(adjacency, smoothness=smoothness)
    else:   
        Q_G = np.identity(d)
    
    # Get Sensor Covariance Components
    AQeA = A @ A.T
    sensor_noise = AQeA / np.trace(AQeA)
    Qe = [sensor_noise,]
    # Get Source Covariance Components
    Qp = []
    LQpL = []
    if inversion_type == 'MSP':
        Ip = np.ceil( np.arange(Np) * d/Np ).astype(int)
        m = deepcopy(len(Ip))  # numer of source cov components
        for i in range(m):
            # First Set
            Q = np.diag(Q_G[:, Ip[i]])
            Qp.append( Q )
            LQpL.append( leadfield@Q@leadfield.T )
            # Extended Set (other hemi)
            j = np.argmin(np.sum(
                np.array([
                    pos[:, 0] + pos[Ip[i], 0],
                    pos[:, 1] - pos[Ip[i], 1], 
                    pos[:, 2] - pos[Ip[i], 2]])**2, axis=0))
            Q = np.diag(Q_G[:, j])
            Qp.append(Q)
            LQpL.append( leadfield@Q@leadfield.T )
            # bilateral
            Q = np.diag(Q_G[:, Ip[i]]) + np.diag(Q_G[:, j])
            Qp.append(Q)
            LQpL.append( leadfield@Q@leadfield.T )
            
    elif inversion_type == 'MNE':
        Qp.append( np.identity(d) )
        LQpL.append( leadfield @ leadfield.T )
            
    elif inversion_type == 'LORETA':
        Qp.append( np.identity(d) )
        LQpL.append( leadfield @ leadfield.T )

        Qp.append( Q_G )
        LQpL.append( leadfield @ Q_G @ leadfield.T )
        
    elif inversion_type == 'BMF':
        inv_cov = np.linalg.inv(C)
        Ns = pos.shape[0]
        allsource = np.zeros(Ns)
        sourcepower = np.zeros(Ns)
        for bk in range(Ns):
            ll = leadfield[:, bk][:, np.newaxis]
            normpower = 1 / (ll.T  @ ll)
            sourcepower[bk] = 1 / (ll.T @  inv_cov @ ll)
            allsource[bk] = sourcepower[bk]  / normpower
        allsource /= allsource.max()
        Qp.append( np.diag(allsource) )
        LQpL.append( leadfield @ np.diag(allsource) @ leadfield.T )

    elif inversion_type == 'BMF-LOR':
        inv_cov = np.linalg.inv(C)
        Ns = pos.shape[0]
        allsource = np.zeros(Ns)
        sourcepower = np.zeros(Ns)
        for bk in range(Ns):
            ll = leadfield[:, bk][:, np.newaxis]
            normpower = 1 / (ll.T  @ ll)
            sourcepower[bk] = 1 / (ll.T @  inv_cov @ ll)
            allsource[bk] = sourcepower[bk]  / normpower
        allsource /= allsource.max()

        Qp.append( np.diag(allsource) )
        LQpL.append( leadfield @ np.diag(allsource) @ leadfield.T )
        
        Qp.append( Q_G )
        LQpL.append( leadfield @ Q_G @ leadfield.T )
        

    m = len(Qp)
    assert len(LQpL) == len(Qp), "Source Covs and projected source covs are not equal."
    

    
    # Source Level Analysis
    # M-STEP
        
    # Combine the noise covariance with the projected source covariances
    Q = np.array(Qe + LQpL)
    # for q in Q:
    #     print(q.shape)
    if inversion_type == "MSP":
        h, _ = spm_sp_reml_demo(C, Q, max_iter)
    else:
        Q0          = np.exp(-2) * np.trace(C) * Qe[0] / np.trace(Qe[0])
        _, h, _, _, _, _ = spm_reml_sc_demo(C, Qe + LQpL, 1, -4, 16, Q0);



    # Ne = len(Qe)
    # Np = len(Qp)
    # # hp = h[Ne + np.arange(Np)]
    # hp = h[Ne:]
    # qp = 0
    # for i in range(Np):
    #     if hp[i] > np.max(hp) / 128:
    #         qp  += hp[i] * Qp[i]
    
    # QP = [np.diag(qp),]
    # LQP = [leadfield @ qp,]
    # LQPL = [LQP[-1] @ leadfield.T]

    # AQ    = AQeA / np.trace(AQeA)
    # Np = len(LQPL)
    # Ne = len(Qe)
    # Q = np.array(Qe + LQPL)

    # Q0 = np.exp(-2) * np.trace(C) * AQ / np.trace(AQ)
    # Cy, h, _, _, _, _ = spm_reml_sc_demo(C, Q, 1, -4, 16, Q0)
    # Cp    = 0
    # LCp   = 0
    # Ne = 1
    # hp    = h[Ne + np.arange(Np)]
    # for j in range(Np):
    #     # print(hp[j], QP[j].shape, LQP[j])
    #     Cp  +=  hp[j] * QP[j]
    #     LCp +=  hp[j] * LQP[j]
    
    
    # M     = LCp.T / Cy 
    
    # E-Step
    sigma_e = sigma(h[1:], Qp)
    sigma_u_lam = sigma(h, Q)
    M = sigma_e @ leadfield.T @ np.linalg.inv(sigma_u_lam)
    
    return M


def spm_sp_reml_demo(YY, Q_c, max_iter):
    n = YY.shape[0]
    m = len(Q_c)
    h = np.zeros(m)  # hyperparameters
    
    
    hE = np.zeros(m) - 32
    hC = np.identity(m) * 256
    hP = np.linalg.inv(hC)

    # Scale
    sY = n*np.trace(YY)
    YY  /= sY

    # find bases of Q if necessary
    Q = []
    for i in range(m):
        q, v, _ = np.linalg.svd(Q_c[i])
        thr = np.exp(-16)
        v *=  v
        keep = (v*len(v) / v.sum()) > thr
        q = q[:, keep]
        v = v[keep]
        C = dict(q=q, v=v)
        Q.append(C)



    # Scaling
    sh = np.zeros(m);
    for i in range(m):
        sh[i] = n * np.trace(Q[i]["q"].T @ Q[i]["q"] @ np.diag(Q[i]["v"]));
        Q[i]["q"]  = Q[i]["q"] / np.sqrt(sh[i]);
    
    q = [qq["q"] for qq in Q]
    v = [qq["v"] for qq in Q]

    a = 0
    for i in range(len(v)):
        a += len(v[i])
    
    dedh = np.zeros((a, len(v)))
    # print("dedh.shape: ", dedh.shape, "v[0].shape: ", v[0].shape, "v[1].shape: ", v[1].shape, )
    dedh[:n, 0] = v[0]
    for i in range(1, dedh.shape[1]):
        b = np.zeros(a)
        b[i+n-1:i+n-1+len(v[i])] = v[i]
        dedh[:, i] = b

    

    delta_F = 1
    a_s = np.arange(m)
    for ii in range(max_iter):
        # C = np.zeros((n, n))
        # e = dedh * np.exp(h)
        # s = len(q)
        # print(len(q), q[0].shape, e.shape)
        # for i in range(s):
        #     C += q[i] @ e[i]

        sigma_u_lam = sigma(h, Q_c)
        sigma_u_pinv = np.linalg.pinv(sigma_u_lam)
        e = np.exp(h)

        W = sigma_u_pinv @ YY @ sigma_u_pinv - sigma_u_pinv
        qr = np.stack([np.diagonal(Q) for Q in Q_c[a_s]], axis=1)
        P = qr.T @ sigma_u_pinv @ qr
        
        dFde = 0.5 * np.sum(qr * (W@qr), axis=0).T
        # print(dFde.shape)
        dFdee = -0.5 * P * P.T

        dhdh = np.diag(np.exp(h[a_s]))
        dFdh = dhdh.T @ dFde
        
        dFdhh = dhdh.T @ dFdee @ dhdh

        e = h - hE
        dFdh = dFdh - hP[a_s, a_s] * e[a_s]
        
        dFdhh = dFdhh - hP[a_s, a_s]

        dh = -np.linalg.pinv(dFdhh) @ dFdh / np.log(ii+3)
        h[a_s] += dh

        dF = dFdh.T @ dh

        # print(f'Iteration {ii+1}. Free Energy Improvement: {dF:.2f}')        
        if (delta_F < 0.01) or (ii == max_iter):
            break
        else:
            a_s = h > -16
            h[~a_s] = hE[~a_s]
            h[h>1] = 1
        
      
    h  = sY*np.exp(h) / sh
    return h, a_s

def sigma(X, Q_e):
        return np.sum([np.e**x * Q for x, Q in zip(X, Q_e)], axis=0)

def spm_dctmtx(N,K):
    # Create basis functions for Discrete Cosine Transform
    # translated and simplified from:
    #  https://github.com/spm/spm12/blob/master/spm_dctmtx.m
    
    n = np.arange(N).T
        

    C = np.zeros((len(n), K))
    # print(np.ones((n.shape[0],1)).shape)
    C[:, 0] = np.ones(n.shape[0]) / np.sqrt(N)

    for k in range(1,K):
        C[:,k] = np.sqrt(2/N)*np.cos(np.pi*(2*n+1)*(k-1)/(2*N))
    return C

def spm_mvb_G_demo(Y,QL,G,Qe):
    # Demo version of the spm_mvb_G_demo() routine
    # Related with the Technical Note:
    # 
    # Algorithmic procedures for Bayesian MEG/EEG source reconstruction in SPM
    # 
    # Created by:	Jose David Lopez - ralph82co@gmail.com
    # 				Gareth Barnes - g.barnes@fil.ion.ucl.ac.uk
    # 				Vladimir Litvak - litvak.vladimir@gmail.com
    # 
    # Date: April 2013


    # 1. INITIALISE DEFAULTS

    # defaults
    Nk    = G.shape[0]
    Np    = G.shape[1]
    Ne    = len(Qe)
    
    # assemble empirical priors
    Qp    = []
    LQpL  = []
    for i in range(Np):
        j = np.where(G[:,i])[0];
        # mat = sparse(j,j,1,Nk,Nk)
        mat = np.zeros((Nk, Nk))
        mat[j, j] = 1
        Qp.append( mat )
        new_mat = np.matmul(np.matmul(QL,mat), QL.T)
        LQpL.append( new_mat ) 
    
    
    # 2. INVERSE SOLUTION
    # Covariance components (with the first error Qe{1} fixed)
    
    if QL.shape[1] > 0:
        Q = [Qe, LQpL]
    else:
        Q = Qe
    
    # ReML (with mildly informative hyperpriors) and a lower bound on error
    N          = Y.shape[1]
    YY         = np.matmul(Y,Y.T)
    Qe         = (np.e**(-2))*np.mamtul(np.trace(YY),Q[0]) / (np.trace(Q[0])/N)
    hE         = -4
    hC         = 16
    # TODO implement
    [Cy,h,Ph,F] = spm_reml_sc_demo(YY,Q,N,hE,hC,Qe)

    # 3. ESTIMATE AND SAVE

    # prior covariance: source space
    Cp    = np.zeros((Nk,Nk))
    for i in range(Np):
        Cp += np.matmul(h[i + Ne], Qp[i])
    
    # MAP estimates of instantaneous sources
    MAP   = np.matmul(Cp,QL.T/Cy)
    qE    = np.matmul(MAP,Y)
    
    
    # assemble results (pattern weights)
    d = dict(F=F, G=G, h=h, qE=qE, MAP=MAP, Cp=Cp)
    return d

def spm_reml_sc_demo(YY,Q,N,hE,hC,Qe):
    # Demo version of the spm_reml_sc_demo() routine
    # Related with the Technical Note:
    #
    # Algorithmic procedures for Bayesian MEG/EEG source reconstruction in SPM
    # 
    # Created by:	Jose David Lopez - ralph82co@gmail.com
    #				Gareth Barnes - g.barnes@fil.ion.ucl.ac.uk
    #				Vladimir Litvak - litvak.vladimir@gmail.com
    #
    # Date: April 2013
    #

    # 1. INITIALISE DEFAULTS

    # initialise h
    n     = Q[0].shape[0]
    m     = len(Q)
    h     = np.zeros(m)
    dFdh  = np.zeros(m)
    dFdhh = np.zeros((m,m))
    
    # check fixed component
    if len(Qe) == 1:
        Qe = np.matmul(Qe,np.eye(n,n))
    
    # initialise and specify hyperpriors
    # scale Q and YY
    sY = np.trace(YY)/(N*n)
    YY = YY / sY
    Qe  = Qe / sY
    sh = np.zeros(m)
    for i in range(m):
        sh[i] = np.trace(Q[i]) / n
        Q[i]    = Q[i]/sh[i]
    

    # hyperpriors
    try:
        hP = np.linalg.inv(hC)
    except:
        hP = 1/hC
    
    
    hE = hE*np.ones(m)

    hP = hP*np.identity(m)

    # intialise h: so that sum(exp(h)) = 1
    if np.any(np.diag(hP) > np.e**(16)):
        h = hE
    
    # 2. ReML (EM/VB)
    dF    = np.inf
    a_s    = np.arange(m)
    t     = 4
    for k in range(32):

        # 2.1 UPDATE PRIOR COVARIANCE MATRIX
        # E-step: conditional covariance cov(B|y) {Cq}
        # compute current estimate of covariance
        C     = Qe
        for i in a_s:
            C += Q[i]*np.exp(h[i])
        P    = np.linalg.inv(C)
        # M-step: ReML estimate of hyperparameters
        # 2.2 COMPUTE GRADIENT AND CURVATURE
        # Gradient dF/dh (first derivatives)
        U     = np.eye(n) - P @ YY / N
        PQ = dict()
        for i in a_s:
            # dF/dh = -trace(dF/diC*iC*Q{i}*iC)
            PQ[i] = P @ Q[i]
            dFdh[i] = -np.trace( PQ[i] @ U) * N/2
        

        # Expected curvature E{dF/dhh} (second derivatives)
        for i in a_s:
            for j in a_s:
                # dF/dhh = -trace{P*Q{i}*P*Q{j}}
                dFdhh[i,j] = -np.trace( PQ[i] @ PQ[j]) * N/2
                dFdhh[j,i] =  dFdhh[i,j]
  
        # 2.3 UPDATE HYPERPARAMETERS
        # modulate
        dFdh  = dFdh * np.exp(h)
        
        dFdhh = np.dot(dFdhh,  np.exp(h)[:, np.newaxis] @ np.exp(h)[np.newaxis])
    
        # add hyperpriors
        e     = h     - hE
        dFdh  = dFdh  - (hP @ e[:, np.newaxis])[:, 0]
        
        dFdhh = dFdhh - hP

        # Fisher scoring: update dh = -inv(ddF/dhh)*dF/dh
        # ToDo: bring "t" into the equation
        dh = -np.linalg.pinv(dFdhh) @ dFdh
        # dh    = spm_dx(dFdhh[a_s,a_s], dFdh[a_s], [t])
        h[a_s] += dh[a_s]

        # predicted change in F - increase regularisation if increasing
        pF    = dFdh[a_s].T @ dh[a_s]
        if pF > dF:
            t -= 1
        else:
            t += 1/8;
        dF    = pF
        
        # print(f"ReML Iteration {k}: {dF}") 
        # print(f'%s %-23d: %10s%e [%+3.2f]\n','  ReML Iteration',k,'...',full(dF),t);
        if dF < 1e-2:
            break
        else:
            # eliminate redundant components (automatic selection)
            a_s  = np.where(h > -16)[0]
            h[~a_s] = hE[~a_s]
            h[a_s] = 1
            # a_s  = a_s.T
        
    # 3. COMPUTATION OF FREE ENERGY

    # log evidence = ln p(y|X,Q) = ReML objective = F = trace(R'*iC*R*YY)/2 ...
    Ph    = -dFdhh
    
    # tr(hP*inv(Ph)) - nh + tr(pP*inv(Pp)) - np (pP = 0)
    # Ft = np.trace(hP/Ph) - len(Ph)
    Ft = 0
    # complexity - KL(Ph,hP)
    # Fc = Ft/2 + np.e*hP*np.e/2 + np.log(np.linalg.det( Ph/hP )) / 2
    Fc = 0
    # Accuracy - ln p(Y|h)
    # Fa = Ft/2 - np.trace(C*P*YY*P)/2 - N*n*np.log(2*np.pi)/2  - N * np.log(np.linalg.det(C))/2
    Fa = 0
    # Free-energy
    # F  = Fa - Fc - N*n*np.log(sY)/2;
    F = 0
    # print('Free-energy: ', F)
    
    

    # return exp(h) hyperpriors and rescale
    # h  = np.log(sY*np.exp(h) / sh)
    # print("final h: ", h)
    C  = sY*C;
    return C,h,Ph,F,Fa,Fc
    