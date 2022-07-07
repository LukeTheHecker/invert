from tabnanny import verbose
import numpy as np
from copy import deepcopy
import mne
from scipy.sparse.csgraph import laplacian

def make_loreta_inverse_operator(leadfield, adjacency, alpha=0.001):
    """ Calculate the inverse operator using Low Resolution TomogrAphy (LORETA).

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    adjacency : numpy.ndarray
        The source adjacency matrix (n_dipoles x n_dipoles) which represents the
        connections of the dipole mesh.
    alpha : float
        The regularization parameter.

    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    B = np.diag(np.linalg.norm(leadfield, axis=0))
    laplace_operator = laplacian(adjacency)
    inverse_operator = np.linalg.inv(leadfield.T @ leadfield + alpha * B @ laplace_operator.T @ laplace_operator @ B) @ leadfield.T
    return inverse_operator


def make_sloreta_inverse_operator(leadfield, alpha=0.001, noise_cov=None):
    """ Calculate the inverse operator using standardized Low Resolution
    TomogrAphy (sLORETA).
    

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    adjacency : numpy.ndarray
        The source adjacency matrix (n_dipoles x n_dipoles) which represents the
        connections of the dipole mesh.
    alpha : float
        The regularization parameter.

    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    n_chans, _ = leadfield.shape
    if noise_cov is None:
        noise_cov = np.identity(n_chans)

    K_MNE = leadfield.T @ np.linalg.inv(leadfield @ leadfield.T + alpha * noise_cov)
    W_diag = 1 / np.diag(K_MNE @ leadfield)

    W_slor = np.diag(W_diag)

    W_slor = np.sqrt(W_slor)

    inverse_operator = W_slor @ K_MNE
    return inverse_operator

def make_eloreta_inverse_operator(leadfield, alpha=0.001, stop_crit=0.005, noise_cov=None,
                                    verbose=0):
    """ Calculate the inverse operator using exact Low Resolution
    TomogrAphy (eLORETA).
    

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    alpha : float
        The regularization parameter.
    stop_crit : float
        Criterium at which to stop optimization of the depth weighting matrix D.
    noise_cov : numpy.ndarray
        The noise covariance matrix (channels x channels).
    verbose : int/bool
        Controls the verbosity of the program.

    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    n_chans, _ = leadfield.shape
    if noise_cov is None:
        noise_cov = np.identity(n_chans)

    # D, C = calc_eloreta_D(leadfield, alpha, stop_crit=stop_crit)
    D = calc_eloreta_D2(leadfield, noise_cov, alpha, stop_crit=stop_crit, verbose=verbose)
    
    
    D_inv = np.linalg.inv(D)
    inverse_operator = D_inv @ leadfield.T @ np.linalg.inv( leadfield @ D_inv @ leadfield.T + alpha * noise_cov )
    
    return inverse_operator


def calc_eloreta_D2(leadfield, noise_cov, alpha, stop_crit=0.005, verbose=0):
    ''' Algorithm that optimizes weight matrix D as described in 
        Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
        https://www.sciencedirect.com/science/article/pii/S1053811920309150
        '''
    n_chans, n_dipoles = leadfield.shape
    # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
    D = np.identity(n_dipoles)

    if verbose>0:
        print('Optimizing eLORETA weight matrix W...')
    cnt = 0
    while True:
        old_D = deepcopy(D)
        if verbose>0:
            print(f'\trep {cnt+1}')
        D_inv = np.linalg.inv(D)
        inner_term = np.linalg.inv(leadfield @ D_inv @ leadfield.T + alpha**2*noise_cov)
            
        for v in range(n_dipoles):
            D[v, v] = np.sqrt( leadfield[:, v].T @ inner_term @ leadfield[:, v] )
        
        averagePercentChange = np.abs(1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D))))
        
        if verbose>0:
            print(f'averagePercentChange={100*averagePercentChange:.2f} %')

        if averagePercentChange < stop_crit:
            if verbose>0:
                print('\t...converged...')
            break
        cnt += 1
    if verbose>0:
        print('\t...done!')
    return D


# def calc_eloreta_D(leadfield, tikhonov, stop_crit=0.005):
#     ''' Algorithm that optimizes weight matrix D as described in 
#         Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
#         https://www.sciencedirect.com/science/article/pii/S1053811920309150
#         '''
#     n_chans, n_dipoles = leadfield.shape
#     # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
#     D = np.identity(n_dipoles)
#     H = centeringMatrix(n_chans)
#     print('Optimizing eLORETA weight matrix W...')
#     cnt = 0
#     while True:
#         old_D = deepcopy(D)
#         print(f'\trep {cnt+1}')
#         C = np.linalg.pinv( np.matmul( np.matmul(leadfield, np.linalg.inv(D)), leadfield.T ) + (tikhonov * H) )
#         for v in range(n_dipoles):
#             leadfield_v = np.expand_dims(leadfield[:, v], axis=1)
#             D[v, v] = np.sqrt( leadfield_v.T @ C @ leadfield_v )
        
#         averagePercentChange = np.abs(1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D))))
#         print(f'averagePercentChange={100*averagePercentChange:.2f} %')
#         if averagePercentChange < stop_crit:
#             print('\t...converged...')
#             break
#         cnt += 1
#     print('\t...done!')
#     return D, C

# def centeringMatrix(n):
#     ''' Centering matrix, which when multiplied with a vector subtract the mean of the vector.'''
#     C = np.identity(n) - (1/n) * np.ones((n, n))
#     return C