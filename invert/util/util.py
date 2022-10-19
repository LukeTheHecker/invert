import mne
import numpy as np

def pos_from_forward(forward, verbose=0):
    ''' Get vertex/dipole positions from mne.Forward model

    Parameters
    ----------
    forward : instance of mne.Forward
        The forward model. 
    
    Return
    ------
    pos : numpy.ndarray
        A 2D matrix containing the MNI coordinates of the vertices/ dipoles

    Note
    ----
    forward must contain some subject id in forward["src"][0]["subject_his_id"]
    in order to work.
    '''
    # Get Subjects ID
    subject_his_id = forward["src"][0]["subject_his_id"]
    src = forward["src"]

    # Extract vertex positions from left and right source space
    pos_left = mne.vertex_to_mni(src[0]["vertno"], 0, subject_his_id, 
                                verbose=verbose)
    pos_right = mne.vertex_to_mni(src[1]["vertno"], 1, subject_his_id, 
                                verbose=verbose)

    # concatenate coordinates from both hemispheres
    pos = np.concatenate([pos_left, pos_right], axis=0)

    return pos

def thresholding(x, k):
    ''' Set all but the k largest magnitudes in x to zero (0).

    Parameters
    ----------
    x : numpy.ndarray
        Data vector
    k : int
        The k number of largest magnitudes to maintain.

    Return
    ------
    x_new : numpy.ndarray
        Array of same length as input array x.
    '''
    if type(x) == list:
        x = np.array(x)
    highest_idc = np.argsort(abs(x))[-k:]
    # print(highest_idc)
    x_new = np.zeros(len(x))
    x_new[highest_idc] = x[highest_idc]
    return x_new

def calc_residual_variance(M_hat, M):
    """ Calculate the residual variance in percent (%).
    
    Parameters
    ----------
    M_hat : numpy.ndarray
        Estimated EEG data
    M : numpy.ndarray
        True EEG data
    
    Return
    ------
    exp_var : float
        Explained variance in %. 
            10 -> 10 % of variance in M are explained by M_hat

    """
    return 100 * np.sum( (M-M_hat)**2 ) / np.sum(M**2)

def euclidean_distance(A, B):
    ''' Euclidean Distance between two points.'''
    return np.sqrt(np.sum((A-B)**2))

def calc_area_tri(AB, AC, CB):
    ''' Calculates area of a triangle given the length of each side.'''
    
    s = (AB + AC + CB) / 2
    area = (s*(s-AB)*(s-AC)*(s-CB)) ** 0.5
    return area

def find_corner(source_power, residual):
    ''' Find the corner of the l-curve given by plotting regularization
    levels (r_vals) against norms of the inverse solutions (l2_norms).

    Parameters
    ----------
    r_vals : list
        Levels of regularization
    l2_norms : list
        L2 norms of the inverse solutions per level of regularization.
    
    Return
    ------
    idx : int
        Index at which the L-Curve has its corner.

    
    '''
    
    # Normalize l2 norms
    # source_power /= np.max(source_power)

    A = np.array([residual[0], source_power[0]])
    C = np.array([residual[-1], source_power[-1]])
    areas = []
    for j in range(1, len(source_power)-1):
        B = np.array([residual[j], source_power[j]])
        AB = euclidean_distance(A, B)
        AC = euclidean_distance(A, C)
        CB = euclidean_distance(C, B)
        area = abs(calc_area_tri(AB, AC, CB))
        areas.append(area)
    if len(areas) > 0:
        idx = np.argmax(areas)+1
    else:
        idx = 0
    return idx