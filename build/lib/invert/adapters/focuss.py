import numpy as np
from copy import deepcopy
import mne

from invert.util.util import pos_from_forward

def focuss(stc, evoked, forward, alpha=0.01, max_iter=10, verbose=0):
    ''' FOCUSS algorithm.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source Estimate object.
    evoked : mne.EvokedArray
        Evoked EEG data object
    forward : mne.Forward
        The forward model
    alpha : float,
        Regularization parameter
    verbose : int
        Controls verbosity of the program
    
    Return
    ------
    stc_focuss : mne.SourceEstimate
        The new focussed source estimate
    '''

    leadfield = forward['sol']['data']
    n_chans, _ = leadfield.shape
    D = stc.data
    M = evoked.data
    w_i = np.diag(1/np.linalg.norm(leadfield, axis=0))
        
    D_FOCUSS = np.zeros(D.shape)
    if verbose:
        print("FOCUSS:\n")
    for t in range(D.shape[1]):
        if verbose > 0:
            print(f"Time step {t}/{D.shape[1]}")
        D_Last = deepcopy(D)
            
        W_i = np.diag(D[:, t])
        
        for i in range(max_iter):
            if verbose > 0:
                print(f'Iteration {i}')
            WWL = W_i @ W_i.T @ leadfield.T
            D_FOCUSS_t = WWL @ np.linalg.inv(leadfield @ WWL + alpha * np.identity(n_chans)) @ M[:, t][:, np.newaxis]
            W_i = w_i @ W_i @ np.diag(D_FOCUSS_t[:, 0])

            if np.linalg.norm(D_FOCUSS_t) == 0:
                D_FOCUSS_t = D_Last
                if verbose:
                    print(f"converged at repetition {i+1}")
                D_FOCUSS[:, t] = D_FOCUSS_t[:, 0]
                break
            else:
                D_Last = deepcopy(D_FOCUSS_t)

    stc_focuss = stc.copy()
    stc_focuss.data = D_FOCUSS
    return stc_focuss

def s_focuss(stc, evoked, forward, alpha=0.01, percentile=0.01, max_iter=10, verbose=0):
    ''' Shrinking FOCUSS algorithm. Based on Grech et al. (2008)

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source Estimate object.
    evoked : mne.EvokedArray
        Evoked EEG data object
    forward : mne.Forward
        The forward model
    alpha : float,
        Regularization parameter
    verbose : int
        Controls verbosity of the program
    
    Return
    ------
    stc_focuss : mne.SourceEstimate
        The new focussed source estimate
    
    References
    ----------
    [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
    Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
    problem in EEG source analysis. Journal of neuroengineering and
    rehabilitation, 5(1), 1-33.

    '''

    leadfield = forward['sol']['data']
    adjacency = mne.spatial_src_adjacency(forward['src'], verbose=verbose).toarray()
    pos = pos_from_forward(forward)
    n_chans, n_dipoles = leadfield.shape
    D = stc.data
    M = evoked.data
    
    w_i = np.diag(1/np.linalg.norm(leadfield, axis=0))
        
    D_FOCUSS = np.zeros(D.shape)
    if verbose:
        print("Shrinking FOCUSS:\n")
    for t in range(D.shape[1]):
        if verbose > 0:
            print(f"Time step {t}/{D.shape[1]}")
        D_Last = deepcopy(D)
        do_smoothing = True
        
        # Initial weighting matrix
        W_i = np.diag(D[:, t])
        WWL = W_i @ W_i.T @ leadfield.T
        sparsities = [1e9,]
        for iter in range(max_iter):
            # if verbose > 0:
                # print(f'\tIteration {iter+1}')
            
            D_FOCUSS_t = WWL @ np.linalg.inv(leadfield @ WWL + alpha * np.identity(n_chans)) @ M[:, t][:, np.newaxis]
            # Smoothing operation (step 4 and 5)
            if do_smoothing:
                D_FOCUSS_t_smooth, sparsity = smooth(D_FOCUSS_t, adjacency, percentile=percentile)
                # leadfield[:, D_FOCUSS_t_smooth[:, 0]==0] *= 0
            else:
                D_FOCUSS_t_smooth = D_FOCUSS_t
            
            
            if do_smoothing:
                if sparsity<n_chans or sparsity>sparsities[-1]:
                    do_smoothing = False
                    # print("\tShrinking is done")
                else:
                    # print(f"sparsity at {sparsity}")
                    sparsities.append(sparsity)
                    D_Last = deepcopy(D_FOCUSS_t_smooth)
                    W_i = w_i @ W_i @ np.diag(D_FOCUSS_t_smooth[:, 0])
            else:
                # Continue with normal FOCUSS
                if np.linalg.norm(D_FOCUSS_t) == 0:
                    D_FOCUSS_t = D_Last
                    # if verbose:
                        # print(f"\tconverged at repetition {iter+1}")
                    D_FOCUSS[:, t] = D_FOCUSS_t[:, 0]
                    break
                else:
                    D_Last = deepcopy(D_FOCUSS_t)
                    W_i = w_i @ W_i @ np.diag(D_FOCUSS_t_smooth[:, 0])
                
                    

    stc_focuss = stc.copy()
    stc_focuss.data = D_FOCUSS
    return stc_focuss

def smooth(D_FOCUSS_t, adjacency, percentile=0.01):

    D_FOCUSS_t_smoothed = deepcopy(D_FOCUSS_t)
    n_dipoles = adjacency.shape[0]

    maximum_value = abs(D_FOCUSS_t).max()
    prominent_idc = np.where(abs(D_FOCUSS_t)>maximum_value*percentile)[0]
    # print(prominent_idc)
    neighbor_list = [np.where(adjacency[idx, :])[0] for idx in prominent_idc]
    neighbor_cat = np.unique(np.concatenate([np.where(adjacency[idx, :]==1)[0] for idx in prominent_idc]))
    non_neighbor_cat = np.unique(np.concatenate([np.where(adjacency[idx, :]==0)[0] for idx in prominent_idc]))

    for i, idx in enumerate(neighbor_cat):
        neighbor_idc = list(np.where(adjacency[idx, :])[0])
        neighbor_idc.remove(idx)
        neighbor_idc = np.array(neighbor_idc)
        D_FOCUSS_t_smoothed[idx] = (D_FOCUSS_t[idx] + D_FOCUSS_t[neighbor_idc].sum()) / (len(neighbor_idc)+1)

    remove_idc = list(np.arange(n_dipoles))
    for neigh in neighbor_cat:
        remove_idc.remove(neigh)
    D_FOCUSS_t_smoothed[remove_idc] = 0
    sparsity = len(prominent_idc)
    return D_FOCUSS_t_smoothed, sparsity