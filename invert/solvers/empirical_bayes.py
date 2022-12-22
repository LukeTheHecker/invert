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

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', max_iter=1000, noise_cov=None, verbose=0, **kwargs):
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
            Maximum number of iterations.
        noise_cov : [None, numpy.ndarray]
            The noise covariance matrix. Use "None" if not available.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.champagne(data, alpha, max_iter=max_iter,)
            inverse_operators.append( inverse_operator )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    
    def champagne(self, y, alpha, max_iter=1000):
        ''' Champagne method.

        Parameters
        ----------
        y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.
        max_iter : int, optional
            The maximum number of inner loop iterations

        Returns
        -------
        x : array, shape (dipoles, time)
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        
        '''
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
        # active_set
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

class SolverEMChampagne(BaseSolver):
    ''' Class for the Expectation Maximization Champagne inverse solution. 

    References
    ----------
    [1] Hashemi, A., & Haufe, S. (2018, September). Improving EEG source
    localization through spatio-temporal sparse Bayesian learning. In 2018 26th
    European Signal Processing Conference (EUSIPCO) (pp. 1935-1939). IEEE.
    '''

    def __init__(self, name="EM-Champagne", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', 
                              max_iter=1000, noise_cov=None, prune=True, 
                              pruning_thresh=1e-3, convergence_criterion=1e-8, 
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
            Maximum number of iterations.
        noise_cov : [None, numpy.ndarray]
            The noise covariance matrix. Use "None" if not available.
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero
            (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov
        self.get_alphas(reference=self.leadfield@self.leadfield.T)
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.em_champagne(data, alpha, max_iter=max_iter, prune=prune, pruning_thresh=pruning_thresh, convergence_criterion=convergence_criterion)
            inverse_operators.append( inverse_operator )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    
    def em_champagne(self, Y, alpha, max_iter=1000, prune=True, 
                          pruning_thresh=1e-3, convergence_criterion=1e-8):
        ''' Expectation Maximization Champagne method.

        Parameters
        ----------
        Y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.
        max_iter : int, optional
            The maximum number of inner loop iterations
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        Returns
        -------
        x : numpy.ndarray
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        
        '''
        n_chans, n_dipoles = self.leadfield.shape
        _, n_times = Y.shape
        L = deepcopy(self.leadfield)
        
        # re-reference data
        Y -= Y.mean(axis=0)

        # Scaling of the data (necessary for convergence criterion and pruning
        # threshold)
        Y_scaled = deepcopy(Y)
        Y_scaled /= abs(Y_scaled).mean()

        I = np.identity(n_chans)
        gammas = np.ones(n_dipoles)
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
        mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
        loss_list = [1e99,]
        for i in range(max_iter):
            old_gammas = deepcopy(gammas)

            for n in range(len(gammas)):
                gammas[n] = Sigma_x[n,n] + (1/n_times) * (mu_x[n]**2).sum()

            gammas[np.isnan(gammas)] = 0
            # print("max gamma: ", gammas.max())
            if prune:
                prune_candidates = gammas<pruning_thresh
                gammas[prune_candidates] = 0
                # print("Pruned: ", prune_candidates.sum())
            # print((gammas==0).sum())
            
            # update rest
            Gamma = np.diag(gammas)
            Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
            Sigma_y_inv = np.linalg.inv(Sigma_y)
            Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
            mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
            loss = np.trace(L@Gamma@L.T) + (1/n_times) * (Y_scaled.T@Sigma_y@Y_scaled).sum()
            loss_list.append(loss)

            # Check if gammas went to zero
            if np.linalg.norm(gammas) == 0:
                # print("breaking")
                gammas = old_gammas
                break
            # Check convergence:
            change = loss_list[-2] - loss_list[-1] 
            # print(change)
            if change < convergence_criterion:
                # print("Converged!")
                break
            
        # update rest
        gammas /= gammas.max()
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma @ L.T @ Sigma_y_inv
        
        # This is how the final source estimate could be calculated:
        # mu_x = inverse_operator @ Y

        return inverse_operator

class SolverMMChampagne(BaseSolver):
    ''' Class for the Majority Maximization Champagne inverse solution. 

    References
    ----------
    [1] Hashemi, A., & Haufe, S. (2018, September). Improving EEG source
    localization through spatio-temporal sparse Bayesian learning. In 2018 26th
    European Signal Processing Conference (EUSIPCO) (pp. 1935-1939). IEEE.
    '''

    def __init__(self, name="MM-Champagne", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', 
                              max_iter=1000, noise_cov=None, prune=True, 
                              pruning_thresh=1e-3, convergence_criterion=1e-8, 
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
            Maximum number of iterations.
        noise_cov : [None, numpy.ndarray]
            The noise covariance matrix. Use "None" if not available.
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero
            (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov
        self.get_alphas(reference=self.leadfield@self.leadfield.T)
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.mm_champagne(data, alpha, max_iter=max_iter, prune=prune, pruning_thresh=pruning_thresh, convergence_criterion=convergence_criterion)
            inverse_operators.append( inverse_operator )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    
    def mm_champagne(self, Y, alpha, max_iter=1000, prune=True, 
                          pruning_thresh=1e-3, convergence_criterion=1e-8):
        ''' Majority Maximization Champagne method.

        Parameters
        ----------
        Y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.
        max_iter : int, optional
            The maximum number of inner loop iterations
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        Returns
        -------
        x : numpy.ndarray
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        
        '''
        n_chans, n_dipoles = self.leadfield.shape
        _, n_times = Y.shape
        L = deepcopy(self.leadfield)
        
        # re-reference data
        Y -= Y.mean(axis=0)

        # Scaling of the data (necessary for convergence criterion and pruning
        # threshold)
        Y_scaled = deepcopy(Y)
        Y_scaled /= abs(Y_scaled).mean()

        I = np.identity(n_chans)
        gammas = np.ones(n_dipoles)
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        # Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
        z_0 = L.T @ Sigma_y_inv @ L
        mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
        loss_list = [1e99,]
        for i in range(max_iter):
            old_gammas = deepcopy(gammas)
            z = []
            for n in range(len(gammas)):
                Ln = L[:, n][:,np.newaxis]
                z_n = Ln.T @ Sigma_y_inv @ Ln
                upper_term = (1/n_times) *(mu_x[n]**2).sum()

                gammas[n] = np.sqrt( upper_term / z_n )
                z.append(z_n)
                # gammas[n] = Sigma_x[n,n] + (1/n_times) * (mu_x[n]**2).sum()
            # z = np.diag(z)
            
            gammas[np.isnan(gammas)] = 0
            # print("max gamma: ", gammas.max())
            if prune:
                prune_candidates = gammas<pruning_thresh
                gammas[prune_candidates] = 0
            
            # update rest
            Gamma = np.diag(gammas)
            Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
            Sigma_y_inv = np.linalg.inv(Sigma_y)
            # Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
            mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
            loss = np.trace(L@Gamma@L.T) + (1/n_times) * (Y_scaled.T@Sigma_y@Y_scaled).sum()
            # first_term = z.T @ gammas
            # loss = first_term + second_term
            loss_list.append(loss)

            # Check if gammas went to zero
            if np.linalg.norm(gammas) == 0:
                # print("breaking")
                gammas = old_gammas
                break
            # Check convergence:
            change = loss_list[-2] - loss_list[-1] 
            # print(change)
            if change < convergence_criterion:
                # print("Converged!")
                break
            
        # update rest
        gammas /= gammas.max()
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma @ L.T @ Sigma_y_inv
        
        # This is how the final source estimate could be calculated:
        # mu_x = inverse_operator @ Y


        return inverse_operator

class SolverTEMChampagne(BaseSolver):
    ''' Class for the Temporal Expectation Maximization Champagne (T-EM
    Champagne) inverse solution. 

    References
    ----------
    [1] Hashemi, A., & Haufe, S. (2018, September). Improving EEG source
    localization through spatio-temporal sparse Bayesian learning. In 2018 26th
    European Signal Processing Conference (EUSIPCO) (pp. 1935-1939). IEEE.
    '''

    def __init__(self, name="TEM-Champagne", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', 
                              max_iter=1000, noise_cov=None, prune=True, 
                              pruning_thresh=1e-3, convergence_criterion=1e-8, 
                              theta=0.01, **kwargs):
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
            Maximum number of iterations.
        noise_cov : [None, numpy.ndarray]
            The noise covariance matrix. Use "None" if not available.
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero
            (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        theta : float
            Another regularization term
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov
        self.get_alphas(reference=self.leadfield@self.leadfield.T)
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.tem_champagne(data, alpha, max_iter=max_iter, prune=prune, pruning_thresh=pruning_thresh, convergence_criterion=convergence_criterion, theta=theta)
            inverse_operators.append( inverse_operator )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def tem_champagne(self, Y, alpha, max_iter=1000, prune=True, 
                          pruning_thresh=1e-3, convergence_criterion=1e-8,
                          theta=0.01):
        ''' Temporal Trade-Off Expectation Maximization Champagne method.

        Parameters
        ----------
        Y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.
        max_iter : int, optional
            The maximum number of inner loop iterations
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        theta : float
            Another regularization term

        Returns
        -------
        x : numpy.ndarray
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        
        '''
        n_chans, n_dipoles = self.leadfield.shape
        _, n_times = Y.shape
        L = deepcopy(self.leadfield)
        
        # re-reference data
        Y -= Y.mean(axis=0)

        # Scaling of the data (necessary for convergence criterion and pruning
        # threshold)
        Y_scaled = deepcopy(Y)
        Y_scaled /= abs(Y_scaled).mean()

        I = np.identity(n_chans)
        It = np.identity(n_times)
        gammas = np.ones(n_dipoles)
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
        mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
        B_hat = np.stack([(mu_x[nn, np.newaxis].T * mu_x[nn, np.newaxis]) / gammas[nn] for nn in range(n_dipoles)], axis=0).sum(axis=0) + theta*It
        B = B_hat / self.frob(B_hat)

        loss_list = [1e99,]
        for i in range(max_iter):
            old_gammas = deepcopy(gammas)

            for n in range(len(gammas)):
                mu_x_n = mu_x[n][np.newaxis]
                gammas[n] = Sigma_x[n,n] + mu_x_n @ np.linalg.inv(B) @ mu_x_n.T

            gammas[np.isnan(gammas)] = 0
            # print("max gamma: ", gammas.max())
            if prune:
                prune_candidates = gammas<pruning_thresh
                gammas[prune_candidates] = 0
                print("Pruned: ", prune_candidates.sum())
            # print((gammas==0).sum())
            
            # update rest
            Gamma = np.diag(gammas)
            Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
            Sigma_y_inv = np.linalg.inv(Sigma_y)
            Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
            mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
            B_hat = np.stack([(mu_x[nn, np.newaxis].T * mu_x[nn, np.newaxis]) / gammas[nn] for nn in range(n_dipoles)], axis=0).sum(axis=0) + theta*It
            B = B_hat / self.frob(B_hat)

            Sigma_0 = np.kron(Gamma, B)
            D = np.kron(L, It)
            Sigma_y_temp = alpha * It + D@Sigma_0@D.T
            y_temp = Y_scaled.reshape(n_chans*n_times, 1)
            loss = np.log(np.linalg.norm(Sigma_y_temp)) + y_temp.T @ np.linalg.inv(Sigma_y_temp) @ y_temp
            # loss = np.trace(L@Gamma@L.T) + (1/n_times) * (Y_scaled.T@Sigma_y@Y_scaled).sum()
            loss_list.append(loss)

            # Check if gammas went to zero
            if np.linalg.norm(gammas) == 0:
                # print("breaking")
                gammas = old_gammas
                break
            # Check convergence:
            change = loss_list[-2] - loss_list[-1] 
            print(change)
            # if change < convergence_criterion:
            #     print("Converged!")
            #     break
            
        # update rest
        gammas /= gammas.max()
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma @ L.T @ Sigma_y_inv
        
        # This is how the final source estimate could be calculated:
        # mu_x = inverse_operator @ Y


        return inverse_operator

    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x@x.T))

class SolverLowSNRChampagne(BaseSolver):
    ''' Class for the LOW SNR Champagne inverse solution. 

    References
    ----------
    [1] Hashemi, A., & Haufe, S. (2018, September). Improving EEG source
    localization through spatio-temporal sparse Bayesian learning. In 2018 26th
    European Signal Processing Conference (EUSIPCO) (pp. 1935-1939). IEEE.
    '''

    def __init__(self, name="LowSNR-Champagne", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, mne_obj, *args, alpha='auto', 
                              max_iter=1000, noise_cov=None, prune=True, 
                              pruning_thresh=1e-3, convergence_criterion=1e-8, 
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
            Maximum number of iterations.
        noise_cov : [None, numpy.ndarray]
            The noise covariance matrix. Use "None" if not available.
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero
            (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        
        Return
        ------
        self : object returns itself for convenience

        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        data = self.unpack_data_obj(mne_obj)

        n_chans = self.leadfield.shape[0]
        if noise_cov is None:
            noise_cov = np.identity(n_chans)
        self.noise_cov = noise_cov
        self.get_alphas(reference=self.leadfield@self.leadfield.T)
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.low_snr_champagne(data, alpha, max_iter=max_iter, prune=prune, pruning_thresh=pruning_thresh, convergence_criterion=convergence_criterion)
            inverse_operators.append( inverse_operator )
        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    
    def low_snr_champagne(self, Y, alpha, max_iter=1000, prune=True, 
                          pruning_thresh=1e-3, convergence_criterion=1e-8):
        ''' Low SNR Champagne method.

        Parameters
        ----------
        Y : array, shape (n_sensors,)
            measurement vector, capturing sensor measurements
        alpha : float
            The regularization parameter.
        max_iter : int, optional
            The maximum number of inner loop iterations
        prune : bool
            If True, the algorithm sets small-activity dipoles to zero (pruning).
        pruning_thresh : float
            The threshold at which small gammas (dipole candidates) are set to
            zero.
        convergence_criterion : float
            Minimum change of loss function until convergence is assumed.
        Returns
        -------
        x : numpy.ndarray
            Parameter vector, e.g., source vector in the context of BSI (x in the cost
            function formula).
        
        '''
        n_chans, n_dipoles = self.leadfield.shape
        _, n_times = Y.shape
        L = deepcopy(self.leadfield)
        
        # re-reference data
        Y -= Y.mean(axis=0)

        # Scaling of the data (necessary for convergence criterion and pruning
        # threshold)
        Y_scaled = deepcopy(Y)
        Y_scaled /= abs(Y_scaled).mean()

        I = np.identity(n_chans)
        gammas = np.ones(n_dipoles)
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        # Sigma_x = Gamma - Gamma @ L.T @ Sigma_y_inv @ L @ Gamma
        mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
        loss_list = [1e99,]
        for i in range(max_iter):
            old_gammas = deepcopy(gammas)

            for n in range(len(gammas)):
                ll = L[:, n][:, np.newaxis]
                LTL = np.diagonal(ll.T@ll)
                gammas[n] = np.sqrt((((mu_x[n]**2).sum()) / n_times) / LTL)

            gammas[np.isnan(gammas)] = 0
            # print("max gamma: ", gammas.max())
            if prune:
                prune_candidates = gammas<pruning_thresh
                gammas[prune_candidates] = 0
                # print("Pruned: ", prune_candidates.sum())
            # print((gammas==0).sum())
            
            # update rest
            Gamma = np.diag(gammas)
            Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
            Sigma_y_inv = np.linalg.inv(Sigma_y)
            mu_x = Gamma @ L.T @ Sigma_y_inv @ Y_scaled
            loss = np.trace(L@Gamma@L.T) + (1/n_times) * (Y_scaled.T@Sigma_y@Y_scaled).sum()
            loss_list.append(loss)

            # Check if gammas went to zero
            if np.linalg.norm(gammas) == 0:
                # print("breaking")
                gammas = old_gammas
                break
            # Check convergence:
            change = loss_list[-2] - loss_list[-1] 
            if change < convergence_criterion:
                # print("Converged!")
                break
            
        # update rest
        gammas /= gammas.max()
        Gamma = np.diag(gammas)
        Sigma_y = (alpha**2) * I + L @ Gamma @ L.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma @ L.T @ Sigma_y_inv
        
        # This is how the final source estimate could be calculated:
        # mu_x = inverse_operator @ Y


        return inverse_operator

