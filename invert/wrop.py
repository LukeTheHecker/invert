import numpy as np
from scipy.spatial.distance import cdist

def make_backus_gilbert_inverse_operator(leadfield, pos):
    """ Calculate the inverse operator using the Backus-Gilbert method.

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    pos : numpy.ndarray
        Position of the vertices in mm


    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    _, n_dipoles = leadfield.shape
  
    
    dist = cdist(pos, pos)

    W_BG = []
    for i in range(n_dipoles):
        W_gamma_BG = np.diag(dist[i, :])
        W_BG.append(W_gamma_BG)

    C = []
    for i in range(n_dipoles):
        C_gamma = leadfield @ W_BG[i] @ leadfield.T
        C.append(C_gamma)

    F = leadfield @ leadfield.T

    E = []
    for i in range(n_dipoles):
        E_gamma = C[i] + F
        E.append(E_gamma)

    L = leadfield @ np.ones((n_dipoles, 1))

    T = []
    for i in range(n_dipoles):
        E_gamma_pinv = np.linalg.pinv(E[i])
        T_gamma = (E_gamma_pinv @ L) / (L.T @ E_gamma_pinv @ L)
        T.append(T_gamma)

    inverse_operator = np.stack(T, axis=0)[:, :, 0]
    
    return inverse_operator


def make_laura_inverse_operator(leadfield, pos, adjacency, alpha=0.001, noise_cov=None,
    drop_off=2):
    """ Calculate the inverse operator using Local AUtoRegressive Average
    (LAURA).

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    pos : numpy.ndarray
        Position of the vertices in mm
    adjacency : numpy.ndarray
        The source adjacency matrix (n_dipoles x n_dipoles) which represents the
        connections of the dipole mesh.
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
    
    d = cdist(pos, pos)
    # Get the adjacency matrix of the source spaces
    for i in range(d.shape[0]):
        # find dipoles that are no neighbor to dipole i
        non_neighbors = np.where(~adjacency.astype(bool)[i, :])[0]
        # append dipole itself
        non_neighbors = np.append(non_neighbors, i)
        # set non-neighbors to zero
        d[i, non_neighbors] = 0
    A = -d**-drop_off
    A[np.isinf(A)] = 0
    W = np.identity(A.shape[0])
    M_j = W @ A

    # Source Space metric
    W_j = np.linalg.inv(M_j.T @ M_j)
    W_j_inv = np.linalg.inv(W_j)

    W_d = np.linalg.inv(noise_cov)
    noise_term = (alpha**2) * np.linalg.inv(W_d)
    inverse_operator = W_j_inv @ leadfield.T @ np.linalg.inv(leadfield @ W_j_inv @ leadfield.T + noise_term)
    return inverse_operator