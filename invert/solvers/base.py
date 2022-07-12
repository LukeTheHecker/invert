import numpy as np
import mne
import matplotlib.pyplot as plt

class InverseOperator:
    ''' This class holds the inverse operator, which may be a simple
    numpy.ndarray matrix or some object like an esinet.net()
    '''
    def __init__(self, inverse_operator, solver_name):
        self.solver_name = solver_name
        self.data = inverse_operator
        self.handle_inverse_operator()
        self.has_multiple_operators()

    def has_multiple_operators(self):
        ''' Check if there are multiple inverse_operators.'''
        if type(self.data) == list:
            if len(self.data) > 1:
                return True
        return False

    def handle_inverse_operator(self):
        if type(self.data) != list:
            self.data = [self.data,]
        self.type = type(self.data[0])
    
    def apply(self, evoked):
        if self.solver_name == "Multiple Sparse Priors" or "bayesian" in self.solver_name.lower():
            M = evoked.data
            
            maximum_a_posteriori, A, S = self.data
            # transform data M with spatial (A) and temporal (S) projector
            M_ = A @ M @ S
            # invert transformed data M_ to tansformed sources J_
            J_ = maximum_a_posteriori @ M_
            # Project tansformed sources J_ back to original time frame using temporal projector S
            return J_ @ S.T 
        elif self.solver_name == "Fully-Connected" or self.solver_name == "Long-Short Term Memory Network" or self.solver_name == "ConvDip":
            net = self.data[0]
            stc = net.predict(evoked)[0]
            J = stc.data
        else:
            M = evoked.data
            J = self.data @ M
            if len(J.shape) > 2:
                J = np.squeeze(J)
        return J
        
        

class BaseSolver:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.r_values = np.insert(np.logspace(-2, 2, 12), 0, 0)
        
    def make_inverse_operator(self, forward: mne.Forward, *args):
        self.forward = forward
        pass

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        ''' Apply the inverse operator '''
        M = evoked.data
        leadfield = self.forward['sol']['data']

        
        source_mats = []
        l2_norms = []
        l2_norms_eeg = []
        l2_residual = []
        # inverse_operators = self.inverse_operator.data
        for inverse_operator in self.inverse_operators:
            # source_mat = inverse_operator.data @ M
            source_mat = inverse_operator.apply( evoked )
            source_mats.append(  source_mat )
            l2_norms.append( np.linalg.norm( source_mat ) )
            l2_norms_eeg.append( np.linalg.norm( leadfield@ source_mat ) )
            l2_residual.append( np.linalg.norm( leadfield@source_mat - M ) )
        if len(self.inverse_operators)>1:
            # Filter non-monotonic decreasing values
            bad_idc = self.filter_norms(self.r_values, l2_norms)
            r_values = np.delete(self.r_values, bad_idc)
            l2_norms = np.delete(l2_norms, bad_idc)
            l2_norms_eeg = np.delete(l2_norms_eeg, bad_idc)
            l2_residual = np.delete(l2_residual, bad_idc)

            corner_idx = self.find_corner(l2_norms)
            source_mat = source_mats[corner_idx]
            # print(f"idx = {corner_idx}, r={r_values[corner_idx]}")

            # plt.figure()
            # plt.subplot(311)            
            # plt.plot(r_values, l2_norms, 'r*')
            # plt.vlines(r_values[corner_idx], ymin=plt.ylim()[0], ymax=plt.ylim()[1])
            # plt.xlabel("R values")
            # plt.ylabel("Norms of the source")

            # plt.subplot(312)
            # plt.plot(r_values[:-1], np.diff(l2_norms), 'r*')
            # plt.vlines(r_values[corner_idx], ymin=plt.ylim()[0], ymax=plt.ylim()[1])
            # plt.xlabel("R values")
            # plt.ylabel("delta Norms of the source")


            # plt.subplot(313)
            # plt.loglog(l2_norms, l2_norms_eeg, 'r*')
            # plt.vlines(l2_norms[corner_idx], ymin=plt.ylim()[0], ymax=plt.ylim()[1])
            # plt.ylabel("Norms of the eeg")
            # plt.xlabel("Norms of the source")

            # plt.figure()
            # plt.loglog(l2_residual, l2_norms, 'r*')
            # plt.vlines(l2_residual[corner_idx], ymin=plt.ylim()[0], ymax=plt.ylim()[1])
            # plt.xlabel("l2_residual")
            # plt.ylabel("l2_norms")
        else:
            source_mat = source_mats[0]
            
        # print(type(source_mat), source_mat.shape)
        stc = self.source_to_object(source_mat, evoked)
        return stc


    def find_corner(self, l2_norms):
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
        l2_norms /= np.max(l2_norms)

        A = np.array([self.r_values[0], l2_norms[0]])
        C = np.array([self.r_values[-1], l2_norms[-1]])
        areas = []
        for j in range(1, len(l2_norms)-1):
            B = np.array([self.r_values[j], l2_norms[j]])
            AB = self.euclidean_distance(A, B)
            AC = self.euclidean_distance(A, C)
            CB = self.euclidean_distance(C, B)
            area = self.calc_area_tri(AB, AC, CB)
            areas.append(area)
        if len(areas) > 0:
            idx = np.argmax(areas)+1
        else:
            idx = 0
        return idx

    @staticmethod
    def filter_norms(r_vals, l2_norms):
        diffs = np.diff(l2_norms)
        bad_idc = []
        all_idc = np.arange(len(l2_norms))
        while np.any( diffs>0 ):
            pop_idx = np.where(diffs>0)[0][0]+1
            r_vals = np.delete(r_vals, pop_idx)
            l2_norms = np.delete(l2_norms, pop_idx)
            diffs = np.diff(l2_norms)
            # print(f"filtered out idx {pop_idx}")
            bad_idc.append(all_idc[pop_idx])
            all_idc = np.delete(all_idc, pop_idx)
        return bad_idc



    @staticmethod
    def euclidean_distance(A, B):
        ''' Euclidean Distance between two points.'''
        return np.sqrt(np.sum((A-B)**2))

    @staticmethod
    def calc_area_tri(AB, AC, CB):
        ''' Calculates area of a triangle given the length of each side.'''
        s = (AB + AC + CB) / 2
        area = (s*(s-AB)*(s-AC)*(s-CB)) ** 0.5
        return area
        
    def source_to_object(self, source_mat, evoked):
        ''' Converts the source_mat matrix to an mne.SourceEstimate object '''
        # Convert source to mne.SourceEstimate object
        source_model = self.forward['src']
        vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
        tmin = evoked.tmin
        sfreq = evoked.info["sfreq"]
        tstep = 1/sfreq
        subject = evoked.info["subject_info"]

        if subject is None:
            subject = "fsaverage"
        
        stc = mne.SourceEstimate(source_mat, vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=self.verbose)
        return stc