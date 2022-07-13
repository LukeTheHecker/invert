import numpy as np
from copy import deepcopy

def focuss(stc, evoked, forward, alpha=0.01, max_iter=10, verbose=0):
    '''
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
            
        W_0 = np.diag(D[:, t])
        W_i = W_0

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