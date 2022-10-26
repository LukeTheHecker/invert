import numpy as np
from copy import deepcopy
import mne
from ..util import calc_residual_variance

def stampc(stc, evoked, forward, max_iter=25, K=1, rv_thresh=0.1, verbose=0):
    ''' Spatio-Temporal Matching Pursuit Contextualizer (STAMP-C)

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source Estimate object.
    evoked : mne.EvokedArray
        Evoked EEG data object
    forward : mne.Forward
        The forward model
    verbose : int
        Controls verbosity of the program
    
    Return
    ------
    stc_focuss : mne.SourceEstimate
        The new focussed source estimate
    '''
    leadfield = forward['sol']['data']
    leadfield_norm = leadfield / np.linalg.norm(leadfield, axis=0)
    
    D = stc.data
    M = evoked.data
    # M = leadfield @ D
    # M -= M.mean(axis=0)
    n_chans, n_dipoles = leadfield.shape
    n_time = M.shape[1]

    # Compute the re-weighting gamma factor from the 
    # existing source estimate
    # Normalize each individual source
    y_hat_model = D / np.linalg.norm(D, axis=0)

    # Compute average source and normalize resulting vector
    gammas_model = np.mean(abs(y_hat_model), axis=1)
    gammas_model /= gammas_model.max()
    
    # Get initial orthogonal leadfield components
    R = deepcopy(M)
    R -= R.mean(axis=0)

    residual_norms = [1e99,]
    idc = np.array([])
    for i in range(max_iter):
        # Calculate leadfield components of the Residual
        mp = (leadfield_norm.T @ R )
        gammas_mp = np.linalg.norm(mp, axis=1, ord=1)
        gammas_mp /= gammas_mp.max()
        
        # Combine leadfield components with the source-gamma
        gammas = gammas_model * gammas_mp
        
        # Select the K dipoles with highest correlation (probability)
        idx = np.argsort(gammas)[-K:]

        # Add the new dipoles to the existing set of dipoles
        idc = np.unique(np.append(idc, idx)).astype(int)

        # Experimental:
        # -------------
        # Remove IDC that dont really explain variance in the whole data
        # Inversion
        # corrs = np.mean(abs((leadfield_norm.T @ D)), axis=1)[idc]
        # highest_K = corrs > corrs.max()/2#np.argsort(corrs)[-K:]
        # idc = idc[highest_K]
        # -------------

        # Inversion: Calculate the inverse solution based on current set
        leadfield_pinv = np.linalg.pinv(leadfield[:, idc])
        y_hat = np.zeros((n_dipoles, n_time))
        y_hat[idc] = leadfield_pinv @ M

        X_hat = leadfield @ y_hat
        # Rereference predicted EEG
        X_hat -= X_hat.mean(axis=0)
        
        # Calculate Residual
        R = M - X_hat
        R -= R.mean(axis=0)

        # Calculate the norm of the EEG-Residual
        residual_norm = np.linalg.norm(R)
        residual_norms.append( residual_norm )
        # Calculate the percentage of residual variance
        rv = calc_residual_variance(X_hat, M)
        print(i, " Res var: ", rv)
        if rv < rv_thresh:
            break

    
    stc_stampc = stc.copy()
    stc_stampc.data = y_hat
    return stc_stampc