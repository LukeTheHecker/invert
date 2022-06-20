import numpy as np
from copy import deepcopy

def calc_eloreta_D(leadfield, tikhonov, stop_crit=0.005):
    ''' Algorithm that optimizes weight matrix D as described in 
        Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
        https://www.sciencedirect.com/science/article/pii/S1053811920309150
        '''
    numberOfElectrodes, numberOfVoxels = leadfield.shape
    # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
    D = np.identity(numberOfVoxels)
    H = centeringMatrix(numberOfElectrodes)
    print('Optimizing eLORETA weight matrix W...')
    cnt = 0
    while True:
        old_D = deepcopy(D)
        print(f'\trep {cnt+1}')
        C = np.linalg.pinv( np.matmul( np.matmul(leadfield, np.linalg.inv(D)), leadfield.T ) + (tikhonov * H) )
        for v in range(numberOfVoxels):
            leadfield_v = np.expand_dims(leadfield[:, v], axis=1)
            D[v, v] = np.sqrt( np.matmul(np.matmul(leadfield_v.T, C), leadfield_v) )
        
        averagePercentChange = np.abs(1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D))))
        print(f'averagePercentChange={100*averagePercentChange:.2f} %')
        if averagePercentChange < stop_crit:
            print('\t...converged...')
            break
        cnt += 1
    print('\t...done!')
    return D, C

def centeringMatrix(n):
    ''' Centering matrix, which when multiplied with a vector subtract the mean of the vector.'''
    C = np.identity(n) - (1/n) * np.ones((n, n))
    return C