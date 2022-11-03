import numpy as np
import mne
from copy import deepcopy
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
                norm_1 = np.linalg.norm(Ps @ l)
                norm_2 = np.linalg.norm(l)
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
                norm_1 = np.linalg.norm(Ps @ l)
                norm_2 = np.linalg.norm(l)
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