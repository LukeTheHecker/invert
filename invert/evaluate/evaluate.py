import numpy as np
import ot
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from copy import deepcopy
from sklearn.metrics import auc, roc_curve
import pandas as pd

def evaluate_all(y_true, y_pred, pos_1, pos_2, argsorted_distance_matrix, distances):
    
    # mse = [eval_mse(yy_true, yy_pred) for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    # nmse = [eval_nmse(yy_true, yy_pred) for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    mle = [eval_mean_localization_error(yy_true, yy_pred, pos_1, pos_2, ghost_thresh=40, threshold=0.01, argsorted_distance_matrix=argsorted_distance_matrix) for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    # auc = [np.mean(eval_auc(yy_true, yy_pred, pos_1, epsilon=0.01, n_redraw=10)) for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    # corr = [pearsonr(yy_true, yy_pred)[0] for yy_true, yy_pred in zip(y_true.T, y_pred.T)]
    emd = eval_emd(distances, abs(y_true).mean(axis=-1), abs(y_pred).mean(axis=-1))
    sparsity_pred = eval_sparsity(y_pred)
    sparsity_true = eval_sparsity(y_true)
    active_true = eval_active(y_true)
    active_pred = eval_active(y_pred)
    
    d = dict(
        # Mean_Squared_Error=np.nanmedian(mse),
        # Normalized_Mean_Squared_Error=np.nanmedian(nmse),
        Mean_Localization_Error=np.nanmedian(mle),
        # AUC=np.nanmedian(auc),
        # Corr=np.nanmedian(corr),
        EMD=emd,
        Sparsity_pred=sparsity_pred,
        Sparsity_true=sparsity_true,
        Active_True=active_true,
        Active_Pred=active_pred,
    )
    
    return d

def eval_active(y, thr=0.01):
    y_norm = np.linalg.norm(y, axis=-1, ord=1)
    y_norm[abs(y_norm)<abs(y_norm).max()*thr] = 0
    return (y_norm!=0).sum() / len(y_norm)

def eval_sparsity(y):
    y_scaled = y / np.linalg.norm(y, axis=0)
    return np.linalg.norm(y_scaled, ord=1)

def eval_emd(M, values_1, values_2):
    values_1 = values_1 / np.sum(values_1)
    values_2 = values_2 / np.sum(values_2)
    emd_value = ot.emd2(values_1, values_2, M)

    return emd_value

def emd(positions_1: np.ndarray, values_1: np.ndarray, positions_2: np.ndarray, values_2: np.ndarray) -> float:
    ''' 
    Compute the Earth Movers Distance between two vectors without a shared grid.
    Parameters:
    - positions_1, positions_2: Lists of vertices' positions for each distribution.
    - values_1, values_2: Magnitudes associated with each vertex position for each distribution.
    '''
    
    # Ensure the sum of the magnitudes for both vectors are equal (normalize if necessary)
    values_1 = values_1 / np.sum(values_1)
    values_2 = values_2 / np.sum(values_2)

    # Calculate pairwise distance between positions
    # M = ot.dist(np.array(positions_1).reshape(-1, 1), np.array(positions_2).reshape(-1, 1))
    M = ot.dist(positions_1, positions_2)
    
    # Compute the EMD
    emd_value = ot.emd2(values_1, values_2, M)
    
    return emd_value

# def eval_emd(distances, distribution1, distribution2):
#     # Convert the data to numpy arrays
#     distribution1 = np.array(distribution1)
#     distribution2 = np.array(distribution2)
    
#     # Normalize the distributions
#     distribution1 = distribution1 / np.sum(distribution1)
#     distribution2 = distribution2 / np.sum(distribution2)
    
#     # Compute the EMD
#     emd = np.sum(distances * np.abs(distribution1 - distribution2))
    
#     return emd

def eval_mse(y_true, y_est):
    '''Returns the mean squared error between predicted and true source. '''
    return np.mean((y_true-y_est)**2)

def eval_nmse(y_true, y_est):
    '''Returns the normalized mean squared error between predicted and true 
    source.'''
    
    y_true_normed = y_true / np.max(np.abs(y_true))
    y_est_normed = y_est / np.max(np.abs(y_est))
    return np.mean((y_true_normed-y_est_normed)**2)


def nmse(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    error = np.zeros(y_true.shape[1])
    for i, (y_true_slice, y_pred_slice) in enumerate(zip(y_true.T, y_pred.T)):
        error[i] = np.mean(((y_true_slice/abs(y_true_slice).max()) - (y_pred_slice/abs(y_pred_slice).max()))**2)
    return error


def corr(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    error = np.zeros(y_true.shape[1])
    for i, (y_true_slice, y_pred_slice) in enumerate(zip(y_true.T, y_pred.T)):
        error[i] = pearsonr(y_true_slice, y_pred_slice)[0]
    return error

def true_variance_explained(y_true, y_pred, leadfield):
    ''' Calculate the true variance explained using the predicted source and the noiseless EEG.
    '''
    M = leadfield @ y_true
    M_hat = leadfield @ y_pred
    # Common average reference
    # M -= M.mean(axis=-1)
    # M_hat -= M_hat.mean(axis=-1)

    return calc_residual_variance(M_hat, M)

def calc_residual_variance(M_hat, M):
    return 100 *  np.sum( (M-M_hat)**2 ) / np.sum(M**2)

def eval_mean_localization_error(y_true, y_est, pos_1, pos_2, k_neighbors=5, 
    min_dist=30, threshold=0.1, ghost_thresh=40, argsorted_distance_matrix=None):
    ''' Calculate the mean localization error for an arbitrary number of 
    sources.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The true source vector (1D)
    y_est : numpy.ndarray
        The estimated source vector (1D)
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int
        The minimum viable distance in mm between maxima. The higher this 
        value, the more maxima will be filtered out.
    ghost_thresh : float/int
        The threshold distance between a true and a predicted source to not 
        belong together anymore. Predicted sources that have no true source 
        within the vicinity defined be ghost_thresh will be labeled 
        ghost_source.
    
    Return
    ------
    mean_localization_error : float
        The mean localization error between all sources in y_true and the 
        closest matches in y_est.
    '''
    if y_est.sum() == 0 or y_true.sum() == 0:
        return np.nan
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)

    
    maxima_true = get_maxima_pos(
        get_maxima_mask(y_true, pos_1, k_neighbors=k_neighbors, 
        threshold=threshold, min_dist=min_dist, 
        argsorted_distance_matrix=argsorted_distance_matrix), pos_1)
    maxima_est = get_maxima_pos(
        get_maxima_mask(y_est, pos_2, k_neighbors=k_neighbors,
        threshold=threshold, min_dist=min_dist, 
        argsorted_distance_matrix=argsorted_distance_matrix), pos_2)

    # Distance matrix between every true and estimated maximum
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:
    closest_matches = distance_matrix.min(axis=1)
    # Filter for ghost sources
    closest_matches = closest_matches[closest_matches<ghost_thresh]
    
    # No source left -> return nan
    if len(closest_matches) == 0:
        return np.nan
    mean_localization_error = np.mean(closest_matches)

    return mean_localization_error


def get_maxima_mask(y, pos, k_neighbors=5, threshold=0.1, min_dist=30,
    argsorted_distance_matrix=None):
    ''' Returns the mask containing the source maxima (binary).
    
    Parameters
    ----------
    y : numpy.ndarray
        The source
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    '''
    if argsorted_distance_matrix is None:
        argsorted_distance_matrix = np.argsort(cdist(pos, pos), axis=1)

    
    y = np.abs(y)
    threshold = threshold*np.max(y)
    # find maxima that surpass the threshold:
    close_idc = argsorted_distance_matrix[:, 1:k_neighbors+1]
    mask = ((y >= np.max(y[close_idc], axis=1)) & (y > threshold)).astype(int)
    

    # filter maxima
    maxima = np.where(mask==1)[0]
    distance_matrix_maxima = cdist(pos[maxima], pos[maxima])
    for i, _ in enumerate(maxima):
        distances_maxima = distance_matrix_maxima[i]
        close_maxima = maxima[np.where(distances_maxima < min_dist)[0]]
        # If there is a larger maximum in the close vicinity->delete maximum
        if np.max(y[close_maxima]) > y[maxima[i]]:
            mask[maxima[i]] = 0
    
    return mask
    
def get_maxima_pos(mask, pos):
    ''' Returns the positions of the maxima within mask.
    Parameters
    ----------
    mask : numpy.ndarray
        The source mask
    pos : numpy.ndarray
        The dipole position matrix
    '''
    return pos[np.where(mask==1)[0]]

def eval_auc(y_true, y_est, pos, n_redraw=25, epsilon=0.25):
    ''' Returns the area under the curve metric between true and predicted
    source. 

    Parameters
    ----------
    y_true : numpy.ndarray
        True source vector 
    y_est : numpy.ndarray
        Estimated source vector 
    pos : numpy.ndarray
        Dipole positions (points x dims)
    n_redraw : int
        Defines how often the negative samples are redrawn.
    epsilon : float
        Defines threshold on which sources are considered
        active.
    Return
    ------
    auc_close : float
        Area under the curve for dipoles close to source.
    auc_far : float
        Area under the curve for dipoles far from source.
    '''
    # Copy
    # t_start = time.time()
    if y_est.sum() == 0 or y_true.sum() == 0:
        return np.nan, np.nan
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    # Absolute values
    y_true = np.abs(y_true)
    y_est = np.abs(y_est)

    # Normalize values
    y_true /= np.max(y_true)
    y_est /= np.max(y_est)

    auc_close = np.zeros((n_redraw))
    auc_far = np.zeros((n_redraw))
    
    # t_prep = time.time()
    # print(f'\tprep took {1000*(t_prep-t_start):.1f} ms')
    
    source_mask = (y_true>epsilon).astype(int)

    numberOfActiveSources = int(np.sum(source_mask))
    # print('numberOfActiveSources: ', numberOfActiveSources)
    numberOfDipoles = pos.shape[0]
    # Draw from the 20% of closest dipoles to sources (~100)
    closeSplit = int(round(numberOfDipoles / 5))
    # Draw from the 50% of furthest dipoles to sources
    farSplit = int(round(numberOfDipoles / 2))
    # t_prep = time.time()
    # print(f'\tprep took {1000*(t_prep-t_start):.1f} ms')

    distSortedIndices = find_indices_close_to_source(source_mask, pos)

    # t_prep2 = time.time()
    # print(f'\tprep2 took {1000*(t_prep2-t_prep):.1f} ms')

    sourceIndices = np.where(source_mask==1)[0]
    
    
  
    
    for n in range(n_redraw):
        
        selectedIndicesClose = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[:closeSplit], size=numberOfActiveSources) ])
        selectedIndicesFar = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[-farSplit:], size=numberOfActiveSources) ])
        # print(f'redraw {n}:\ny_true={y_true[selectedIndicesClose]}\y_est={y_est[selectedIndicesClose]}')
        fpr_close, tpr_close, _ = roc_curve(source_mask[selectedIndicesClose], y_est[selectedIndicesClose])
   
        fpr_far, tpr_far, _  = roc_curve(source_mask[selectedIndicesFar], y_est[selectedIndicesFar])
        
        auc_close[n] = auc(fpr_close, tpr_close)
        auc_far[n] = auc(fpr_far, tpr_far)
    
    auc_far = np.mean(auc_far)
    auc_close = np.mean(auc_close)

  
 
    

    return np.mean([auc_close, auc_far])

def find_indices_close_to_source(source_mask, pos):
    ''' Finds the dipole indices that are closest to the active sources. 

    Parameters
    -----------
    simSettings : dict
        retrieved from the simulate_source function
    pos : numpy.ndarray
        list of all dipole positions in XYZ coordinates

    Return
    -------
    ordered_indices : numpy.ndarray
        ordered list of dipoles that are near active 
        sources in ascending order with respect to their distance to the next source.
    '''

    numberOfDipoles = pos.shape[0]

    sourceIndices = np.array([i[0] for i in np.argwhere(source_mask==1)])
    
    min_distance_to_source = np.zeros((numberOfDipoles))
    
    
    # D = np.zeros((numberOfDipoles, len(sourceIndices)))
    # for i, idx in enumerate(sourceIndices):
    #     D[:, i] = np.sqrt(np.sum(((pos-pos[idx])**2), axis=1))
    # min_distance_to_source = np.min(D, axis=1)
    # min_distance_to_source[source_mask==1] = np.nan
    # numberOfNans = source_mask.sum()
    
    ###OLD
    numberOfNans = 0
    for i in range(numberOfDipoles):
        if source_mask[i] == 1:
            min_distance_to_source[i] = np.nan
            numberOfNans +=1
        elif source_mask[i] == 0:
            distances = np.sqrt(np.sum((pos[sourceIndices, :] - pos[i, :])**2, axis=1))
            min_distance_to_source[i] = np.min(distances)
        else:
            print('source mask has invalid entries')
    # print('new: ', np.nanmean(min_distance_to_source), min_distance_to_source.shape)
    ###OLD

    ordered_indices = np.argsort(min_distance_to_source)

    return ordered_indices[:-numberOfNans]
