import numpy as np
import mne
from copy import deepcopy
from .base import BaseSolver, InverseOperator

class SolverBCS(BaseSolver):
    ''' Class for the Bayesian Compressed Sensing (BCS) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Bayesian Compressed Sensing", **kwargs):
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
        # self.leadfield_norm = (self.leadfield.T / np.linalg.norm(self.leadfield, axis=1)).T
        # self.leadfield_norm = self.leadfield / np.linalg.norm(self.leadfield, axis=0)
        self.leadfield_norm = self.leadfield
        return self

    def apply_inverse_operator(self, evoked, max_iter=100, alpha_0=0.01, eps=1e-16) -> mne.SourceEstimate:
        source_mat = self.calc_bcs_solution(evoked, max_iter=max_iter, alpha_0=alpha_0, eps=eps)
        stc = self.source_to_object(source_mat, evoked)
        return stc
    
    def calc_bcs_solution(self, evoked, max_iter=100, alpha_0=0.01, eps=1e-16):

        alpha_0 = np.clip(alpha_0, a_min=1e-6, a_max=None)
        y = evoked.data
        n_chans, _ = y.shape
        n_dipoles = self.leadfield_norm.shape[1]
        
        # preprocessing
        y -= y.mean(axis=0)
        
        alphas = np.ones(n_dipoles)
        D = np.diag(alphas)

        LLT = self.leadfield_norm.T @ self.leadfield_norm
        sigma = np.linalg.inv(alpha_0 * LLT + D)
        mu = alpha_0 * sigma @ self.leadfield_norm.T @ y
        proj_norm = self.leadfield_norm.T @ y
        proj = self.leadfield.T @ y

        # D_inv = np.linalg.inv(D)
        # var = 1/alpha_0
        # I = np.identity(n_chans)
        # C = var**2 * I + self.leadfield @ D_inv @ self.leadfield.T
        # marginal_likelihood = -0.5 * (n_chans * np.log(2*np.pi) + np.log(C) + y.T @ np.linalg.inv(C) @ y)
        
        residual_norms = [1e99]
        x_hats = []
        for i in range(max_iter):
            gammas = np.array([1 - alphas[ii] * sigma[ii,ii] for ii in range(n_dipoles)])
            gammas[np.isnan(gammas)] = 0

            
            alphas = gammas / np.linalg.norm(mu**2, axis=1)
            alpha_0 = 1 / ( np.linalg.norm(y - self.leadfield_norm @ mu) / (n_chans - gammas.sum()) )
            D = np.diag(alphas) + eps
            sigma = np.linalg.inv(alpha_0 * LLT + D)
            mu = alpha_0 * sigma @ proj_norm

            # var = 1/alpha_0
            # try:
            #     D_inv = np.linalg.inv(D)
            # except:
            #     break
            # C = var**2 * I + self.leadfield @ D_inv @ self.leadfield.T
            
            Gamma = np.diag(gammas)
            x_hat = Gamma @ proj
            residual_norm = np.linalg.norm(y - self.leadfield @ x_hat)
            # print(residual_norm)
            if residual_norm > residual_norms[-1]:
                x_hat = x_hats[-1]
                # print(f"Stopping after {i} iterations.")
                break
            residual_norms.append(residual_norm)
            x_hats.append(x_hat)
            # print("residual_norm: ", residual_norm)

        return x_hat