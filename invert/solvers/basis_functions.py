import numpy as np
import mne
from copy import deepcopy
from scipy.sparse import coo_matrix
from .base import BaseSolver, InverseOperator
from ..util import pos_from_forward

class SolverBasisFunctions(BaseSolver):
    ''' Class for the Minimum Norm Estimate (MNE) inverse solution [1] using
    basis functions. Gemoetric informed basis functions are based on [2].

    Attributes
    ----------
    
    References
    ----------
    [1] Pascual-Marqui, R. D. (1999). Review of methods for solving the EEG
    inverse problem. International journal of bioelectromagnetism, 1(1), 75-86.
    
    [2] Wang, S., Wei, C., Lou, K., Gu, D., & Liu, Q. (2024). Advancing EEG/MEG
    Source Imaging with Geometric-Informed Basis Functions. arXiv preprint
    arXiv:2401.17939.

    '''
    def __init__(self, name="Minimum Norm Estimate with Basis Functions", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, function="GBF", alpha="auto", verbose=0, **kwargs):
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
        # n_chans, _ = leadfield.shape

        self.get_inverse_operator = self.create_basis_function(function)
        
        # No regularization leads to weird results with this approach
        if 0 in self.alphas and len(self.alphas) > 1:
            idx = self.alphas.index(0)
            self.alphas.pop(idx)
            self.r_values = np.delete(self.r_values, idx)
        elif 0 in self.alphas and len(self.alphas) == 1:
            idx = self.alphas.index(0)
            self.alphas = [0.01]

        inverse_operators = []
        for alpha in self.alphas:
            inverse_operator = self.get_inverse_operator(alpha)
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [InverseOperator(inverse_operator, self.name) for inverse_operator in inverse_operators]
        return self
    
    def create_basis_function(self, function="GBF"):
        if function.lower() == "gbf":
            return self.create_gbf()
        else:
            raise ValueError(f"Function {function} not implemented.")
    
    def create_gbf(self):
        ''' Create geometric informed basis functions. '''
        n_vertices_left = self.forward['src'][0]['nuse']
        self.faces = np.concatenate([
            self.forward['src'][0]['use_tris'],
            n_vertices_left + self.forward['src'][1]['use_tris'],
        ], axis=0)

        # self.pos = np.concatenate([
        #     self.forward['src'][0]['rr'],
        #     self.forward['src'][1]['rr'],
        # ], axis=0)
        self.pos = pos_from_forward(self.forward)
        
        A = self.compute_laplace_beltrami(
            self.pos.T, self.faces
            )
        _, eigenvalues, _ = np.linalg.svd(A.toarray(), full_matrices=False)
        Sigma = np.diag(1 / (eigenvalues + 0.1 * np.mean(eigenvalues)))
        Sigma_inv = np.linalg.inv(Sigma)
        L = self.leadfield @ A

        return lambda alpha: np.linalg.inv(L.T @ L + alpha * Sigma_inv) @ L.T
    
    @staticmethod
    def cotangent_weight(v1, v2, v3):
        # Compute the cotangent weight of the edge opposite to v1
        edge1 = v2 - v1
        edge2 = v3 - v1
        cotangent = np.dot(edge1, edge2) / np.linalg.norm(np.cross(edge1, edge2))
        return cotangent

    
    def compute_laplace_beltrami(self, pos, faces):
        n = pos.shape[1]  # Number of vertices
        I = []
        J = []
        V = []
        
        for face in faces:
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                
                vi = pos[:, face[i]]
                vj = pos[:, face[j]]
                vk = pos[:, face[k]]
                
                # Compute cotangent weights for edges (vi, vj) and (vi, vk)
                cot_jk = self.cotangent_weight(vi, vj, vk)
                cot_kj = self.cotangent_weight(vi, vk, vj)
                
                # Update the entries for the Laplacian matrix
                I.append(face[i])
                J.append(face[j])
                V.append(-0.5 * (cot_jk + cot_kj))
                
                # Add the contribution to the diagonal element
                I.append(face[i])
                J.append(face[i])
                V.append(0.5 * (cot_jk + cot_kj))
        
        # Create the sparse Laplacian matrix
        L = coo_matrix((V, (I, J)), shape=(n, n))
        
        return L