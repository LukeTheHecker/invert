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

class SolverGammaMAP(BaseSolver):
    ''' Class for the Gamma Maximum A Posteriori (Gamma-MAP) inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Gamma MAP", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha="auto", 
                              max_iter=100, verbose=0, **kwargs):
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
        
        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.make_gamma_map_inverse_operator(evoked.data, alpha, max_iter=max_iter)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)
    
    def make_gamma_map_inverse_operator(self, B, alpha, max_iter=100):
        L = deepcopy(self.leadfield)
        db, n = B.shape
        ds = L.shape[1]

        # Ensure Common average reference
        B -= B.mean(axis=0)
        L -= L.mean(axis=0)
        
 
        # Data Covariance Matrix
        # Cb = B @ B.T
        gammas = np.ones(ds)
        sigma_e = alpha * np.identity(db)  
        sigma_s = np.identity(ds) # identity leads to weighted minimum L2 Norm-type solution
        sigma_b = sigma_e + L @ sigma_s @ L.T
        sigma_b_inv = np.linalg.inv(sigma_b)
        
        for k in range(max_iter):
            # print(k)
            old_gammas = deepcopy(gammas)
            # E = sigma_s @ L.T @ np.linalg.inv( sigma_e + L @ sigma_s @ L.T ) @ B
            # term_1 = sigma_s @ L.T
            # term_2 = np.linalg.inv(sigma_e + L@sigma_s@L.T) @ L @ sigma_s
            # Cov = sigma_s - term_1 @ term_2
            

            term_1 = (gammas/np.sqrt(n)) * np.sqrt(np.sum((L.T @ sigma_b_inv @ B )**2, axis=1))
            term_2 = 1 / np.diagonal(np.sqrt((L.T @ sigma_b_inv @ L )))
            gammas = term_1 * term_2
            if np.linalg.norm(gammas) == 0:
                gammas = old_gammas
                break
            # gammas /= np.linalg.norm(gammas)

        gammas_final = gammas / gammas.max()
        sigma_s_hat = np.diag(gammas_final) @ sigma_s  #  np.array([gammas_final[i] * C[i] for i in range(ds)])
        inverse_operator = sigma_s_hat @ L.T @ np.linalg.inv(sigma_e + L @ sigma_s_hat @ L.T)
        # S = inverse_operator @ B
        return inverse_operator
        
    @staticmethod
    def frob(x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.sqrt(np.trace(x@x.T))