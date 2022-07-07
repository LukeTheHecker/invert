import numpy as np
from copy import deepcopy

def focuss(D, M, leadfield, alpha, verbose=0):
    '''
    Parameters
    ----------
    D : numpy.ndarray
        Initial source estimate
    M : numpy.ndarray
        EEG data 
    leadfield : numpy.ndarray
        The leadfield (or gain matrix)
    
    Return
    ------
    D_FOCUSS : numpy.ndarray
        The new focussed source estimate
    '''
    n_chans, n_dipoles = leadfield.shape
    D_FOCUSS = np.zeros(D.shape)
    if verbose:
        print("FOCUSS:\n")
    for t in range(D.shape[1]):
        i = 0
        # Start by initializing W_0
        
        D_Last = deepcopy(D)
            
        W_0 = np.diag(D[:, t])
        w_i = np.diag(1/np.linalg.norm(leadfield, axis=0))
        W_i = W_0

        while True:
            if verbose:
                print(f'Iteration {i}')
            D_FOCUSS_t = W_i @ W_i.T @ leadfield.T @ np.linalg.inv(leadfield @ W_i @ W_i.T @ leadfield.T + alpha * np.identity(n_chans)) @ M[:, t][:, np.newaxis]
            W_i = w_i @ W_i @ np.diag(D_FOCUSS_t[:, 0])
            if np.linalg.norm(D_FOCUSS_t) == 0:
                D_FOCUSS_t = D_Last
                if verbose:
                    print(f"converged at repetition {i+1}")
                D_FOCUSS[:, t] = D_FOCUSS_t[:, 0]
                break
            else:
                D_Last = deepcopy(D_FOCUSS_t)
            i += 1
    return D_FOCUSS