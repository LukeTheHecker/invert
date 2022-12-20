import numpy as np
from copy import deepcopy
import mne
from scipy.sparse.csgraph import laplacian
from ..util import calc_residual_variance

def stampc(stc, evoked, forward, max_iter=25, K=1, rv_thresh=0.1, 
            n_orders=0, verbose=0):
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
    D = stc.data
    M = evoked.data
    # M = leadfield @ D
    # M -= M.mean(axis=0)
    n_dipoles, n_time = D.shape
    
    leadfield = forward['sol']['data']
    leadfield -= leadfield.mean(axis=0)
    leadfield_norm = leadfield / np.linalg.norm(leadfield, axis=0)

    leadfields_norm = [leadfield_norm, ]
    neighbors_bases = [np.arange(n_dipoles),]
    
    # Compute Leadfield bases and corresponding neighbors
    adjacency = mne.spatial_src_adjacency(forward['src'], verbose=0)
    for order in range(n_orders):
        laplace_operator = laplacian(adjacency)
        laplace_operator = laplace_operator

        leadfield_smooth = leadfield @ abs(laplace_operator)
        leadfield_smooth -= leadfield_smooth.mean(axis=0)
        leadfield_smooth_norm = leadfield_smooth / np.linalg.norm(leadfield_smooth, axis=0)

        neighbors_base = [np.where(adj !=0)[0] for adj in adjacency.toarray()]

        leadfields_norm.append( leadfield_smooth_norm )
        neighbors_bases.append( neighbors_base )
        
        adjacency = adjacency @ adjacency.T

    
    
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
        # Calculate leadfield components of the largest eigenvector of the
        # Residual
        sigma_R = R@R.T
        U, _, _ = np.linalg.svd(sigma_R, full_matrices=False)
        
        # Select Gammas of Matching pursuit using the orthogonal leadfield:
        # gammas_mp = abs(leadfield_norm.T @ U[:, 0] )

        # Select the most informativ basis of Gammas
        gammas_bases = [abs(L_norm.T @ U[:, 0]) for L_norm in leadfields_norm]
        basis_idx = np.argmax([gammas_base.max() for gammas_base in gammas_bases])
        # basis_idx = np.argmax([np.mean(gammas_base) for gammas_base in gammas_bases])
        gammas_mp = gammas_bases[basis_idx]
        neighbors_base = neighbors_bases[basis_idx]

        gammas_mp /= gammas_mp.max()
        

        # Combine leadfield components with the source-gamma
        gammas = gammas_model * gammas_mp
        # gammas = gammas_mp + (gammas_model * gammas_mp)
        # gammas = gammas_mp
 
        # Select the K dipoles with highest correlation (probability)
        idx = np.argsort(gammas)[-K:]
        idx = [neighbors_base[idxx] for idxx in idx]

        # Add the new dipoles to the existing set of dipoles
        idc = np.unique(np.append(idc, idx)).astype(int)

        # Inversion: Calculate the inverse solution based on current set
        # V1
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
        # print(i, " Res var: ", round(rv, 2))
        if rv < rv_thresh or residual_norms[-2] == residual_norms[-1]:
            break

    
    stc_stampc = stc.copy()
    stc_stampc.data = y_hat
    return stc_stampc