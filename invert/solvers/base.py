from copy import deepcopy
import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import pickle as pkl
import tensorflow as tf
from mne.io.constants import FIFF
    
class InverseOperator:
    ''' This class holds the inverse operator, which may be a simple
    numpy.ndarray matrix or some object like an esinet.net()

    Parameters
    ----------
    inverse operator : 
    Return
    ------


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
    
    def apply(self, M):
        ''' Apply the precomputed inverse operator to the data matrix M.
        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)

        Return
        ------
        J : numpy.ndarray
            The source estimate matrix (n_sources, n_timepoints)
        '''
 
        J = self.data @ M
        if len(J.shape) > 2:
            J = np.squeeze(J)
        return J
        
        

class BaseSolver:
    '''
    Parameters
    ----------
    regularisation_method : str
        Can be either 
            "GCV"       -> generalized cross validation
            "L"         -> L-Curve method using triangle method
            "L_new"     -> L-Curve method using triangle and scale-free params
            "Product"   -> Minimal product method

    n_reg_params : int
        The number of regularisation parameters to try. The higher, the 
        more accurate the regularisation and the slower the computations.
    prep_leadfield : bool
        If True -> Apply common average referencing on the leadfield columns.
    '''
    def __init__(self, regularisation_method="GCV", n_reg_params=24, 
        prep_leadfield=True, use_last_alpha=False, verbose=0):
        self.verbose = verbose
        # self.r_values = np.insert(np.logspace(-3, 3, n_reg_params), 0, 0)
        self.r_values = np.insert(np.logspace(-3, 3, n_reg_params), 0, 0)


        # self.alphas = deepcopy(self.r_values)
        self.n_reg_params = n_reg_params
        self.regularisation_method = regularisation_method
        self.prep_leadfield = prep_leadfield
        self.use_last_alpha = use_last_alpha
        self.last_reg_idx = None
        
    def make_inverse_operator(self, forward: mne.Forward, *args, alpha="auto", **kwargs):
        """ Base function to create the inverse operator based on the forward
            model.

        Parameters
        ----------
        forward : mne.Forward
            The mne Forward model object.
        alpha : ["auto", float]
            If "auto": Try out multiple regularization parameters and choose the
            optimal one, otherwise use the float.


        Return
        ------
        None

        """
        self.forward = forward
        self.prepare_forward()
        self.alpha = alpha
        self.alphas = self.get_alphas()

    def store_obj_information(self, mne_obj):
        if hasattr(mne_obj, "tmin"):
            self.tmin = mne_obj.tmin
        else:
            self.tmin = 0
        
        self.obj_info = mne_obj.info
        

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        ''' Apply the inverse operator
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne SourceEstimate object.
        
        '''
        
        data = self.unpack_data_obj(mne_obj)
        

        if len(self.inverse_operators) == 1:
            source_mat = self.inverse_operators[0].apply( data )
        elif self.use_last_alpha and self.last_reg_idx is not None:
            source_mat = self.inverse_operators[self.last_reg_idx].apply( data ) 
            
        else:
            if self.regularisation_method.lower() == "l":
                source_mat, idx = self.regularise_lcurve(data)
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() == "gcv":
                source_mat, idx = self.regularise_gcv(data)
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() == "product":
                source_mat, idx = self.regularise_product(data)
                self.last_reg_idx = idx
            else:
                msg = f"{self.regularisation_method} is no valid regularisation method."
                raise AttributeError(msg)
            
            
        # print(type(source_mat), source_mat.shape)
        stc = self.source_to_object(source_mat)
        return stc
        
    @staticmethod
    def prep_data(mne_obj):
        if not mne_obj.proj:
            mne_obj.set_eeg_reference("average", projection=True, verbose=0).apply_proj(verbose=0)
        
        return mne_obj

    def unpack_data_obj(self, mne_obj, pick_types=None):
        ''' Unpacks the mne data object and returns the data.

        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.EvokedArray, mne.Epochs, mne.EpochsArray, mne.Raw]

        Return
        ------
        data : numpy.ndarray
            The M/EEG data matrix.

        '''

        type_list = [mne.Evoked, mne.EvokedArray, mne.Epochs, mne.EpochsArray, mne.io.Raw, mne.io.RawArray]
        if pick_types is None:
            pick_types = dict(meg=True, eeg=True, fnirs=True)
        
        # Prepare Data
        mne_obj = self.prep_data(mne_obj)

        channels_in_fwd = self.forward.ch_names
        channels_in_mne_obj = mne_obj.ch_names
        picks = self.select_list_intersection(channels_in_fwd, channels_in_mne_obj)
        
        # Select only data channels in mne_obj
        mne_obj_meeg = mne_obj.copy().pick_channels(picks).pick_types(**pick_types)
        
        # Store original forward model for later
        self.forward_original = deepcopy(self.forward)

        # Select only available data channels in forward
        self.forward = self.forward.pick_channels(picks)
        
        # Prepare the potentially new forward model
        self.prepare_forward()

        # Test if ch_names in forward model and mne_obj_meeg are equal
        assert self.forward.ch_names == mne_obj_meeg.ch_names, "channels available in mne object are not equal to those present in the forward model."
        assert len(self.forward.ch_names) > 1, "forward model contains only a single channel"

        # check if the object is an evoked object
        if isinstance(mne_obj, (mne.Evoked, mne.EvokedArray)):
            # handle evoked object
            data = mne_obj_meeg.data
        
        # check if the object is a raw object
        elif isinstance(mne_obj, (mne.Epochs, mne.EpochsArray)):
            data = mne_obj_meeg.average().data
        
        # check if the object is a raw object
        elif isinstance(mne_obj, (mne.io.Raw, mne.io.RawArray)):
            # handle raw object
            data = mne_obj_meeg._data
            # data = mne_obj_meeg.get_data()

        # handle other cases
        else:
            msg = f"mne_obj is of type {type(mne_obj)} but needs to be one of the following types: {type_list}"
            raise AttributeError(msg)
        
        self.store_obj_information(mne_obj)
        
        return data
    
    @staticmethod
    def select_list_intersection(list1, list2):
        new_list = []
        for element in list1:
            if element in list2:
                new_list.append(element)
        return new_list

    def get_alphas(self, reference=None):
        ''' Create list of regularization parameters (alphas) based on the
        largest eigenvalue of the leadfield or some reference matrix.

        Parameters
        ----------
        reference : [None, numpy.ndarray]
            If None: use leadfield to calculate regularization parameters, else
            use reference matrix (e.g., M/EEG covariance matrix).
        
        Return
        ------
        alphas : list
            List of regularization parameters (alphas)

        '''
        if reference is None:
            _, eigs, _ = np.linalg.svd(self.leadfield) 
        else:
            _, eigs, _ = np.linalg.svd(reference)
        self.max_eig = eigs.max()

        if self.alpha == "auto":
            
            alphas = list(self.max_eig * self.r_values)
        else:
            alphas = [self.alpha*self.max_eig, ]
        return alphas

    def regularise_lcurve(self, M):
        """ Find optimally regularized inverse solution using the L-Curve method [1].
        
        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)
        
        Return
        ------
        source_mat : numpy.ndarray
            The inverse solution  (dipoles x time points)
        optimum_idx : int
            The index of the selected (optimal) regularization parameter
        
        References
        ----------
        [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
        Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
        problem in EEG source analysis. Journal of neuroengineering and
        rehabilitation, 5(1), 1-33.
        
        """

        leadfield = self.leadfield
        source_mats = [inverse_operator.apply( M ) for inverse_operator in self.inverse_operators]
        
        M -= M.mean(axis=0)
        leadfield -= leadfield.mean(axis=0)
        

        # l2_norms = [np.log(np.linalg.norm( leadfield @ source_mat )) for source_mat in source_mats]
        # l2_norms = [np.log(np.linalg.norm(source_mat )) for source_mat in source_mats]
        l2_norms = [np.linalg.norm( source_mat ) for source_mat in source_mats]
        
        
        # residual_norms = [np.log(np.linalg.norm( leadfield @ source_mat - M )) for source_mat in source_mats]
        residual_norms = [np.linalg.norm( leadfield @ source_mat - M ) for source_mat in source_mats]



        # Filter non-monotonic decreasing values
        # bad_idc = self.filter_norms(self.r_values, l2_norms)
        # l2_norms = np.delete(l2_norms, bad_idc)
        # source_mats = self.delete_from_list(source_mats, bad_idc)
        
        optimum_idx = self.find_corner(l2_norms, residual_norms)
        
        # curvature = self.get_curvature(residual_norms, l2_norms)
        # print(curvature)
        # optimum_idx = np.argmax(curvature)


        source_mat = source_mats[optimum_idx]
        
        # plt.figure()
        # plt.plot(residual_norms, l2_norms, 'ok')
        # plt.plot(residual_norms[optimum_idx], l2_norms[optimum_idx], 'r*')
        # alpha = self.alphas[optimum_idx]
        # plt.title(f"L-Curve: {alpha}")

        return source_mat, optimum_idx
        
    @staticmethod
    def get_curvature(x, y):
        
        x_t = np.gradient(x)
        y_t = np.gradient(y)
        vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])
        speed = np.sqrt(x_t * x_t + y_t * y_t)
        tangent = np.array([1/speed] * 2).transpose() * vel

        ss_t = np.gradient(speed)
        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)

        curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5

        return curvature_val

    def regularise_gcv(self, M):
        """ Find optimally regularized inverse solution using the generalized
        cross-validation method [1].
        
        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)
        
        Return
        ------
        source_mat : numpy.ndarray
            The inverse solution  (dipoles x time points)
        optimum_idx : int
            The index of the selected (optimal) regularization parameter
        
        References
        ----------
        [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
        Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
        problem in EEG source analysis. Journal of neuroengineering and
        rehabilitation, 5(1), 1-33.

        """
        n_chans = self.leadfield.shape[0]
        # Common Average Reference
        M -= M.mean(axis=0)
        
        I = np.identity(n_chans)
        gcv_values = []
        for inverse_operator in self.inverse_operators:
            x = inverse_operator.data @ M
            M_hat = self.leadfield @ x 
            # M_hat -= M_hat.mean(axis=0)
            residual_norm = np.linalg.norm(M_hat- M)
            denom = np.trace(I - self.leadfield @ inverse_operator.data[0])**2
    
            gcv_value = residual_norm / denom
            gcv_values.append(gcv_value)
            # print(np.linalg.norm(x), gcv_value)
        # Filter gcv_values that first increase
        if len(np.where((np.diff(gcv_values)<0))[0]) == 0:
            if not np.isnan(gcv_values[0]):
                keep_idx = 0
            else:
                print("can you read this")
                keep_idx = 1
        else:
            keep_idx = np.where((np.diff(gcv_values)<0))[0][0]
            
        optimum_idx = np.argmin(gcv_values[keep_idx:])+keep_idx

        # optimum_idx = np.argmin(gcv_values[1:])+1

        # plt.figure()
        # plt.loglog(self.alphas, gcv_values, 'ok')
        # plt.plot(self.alphas[optimum_idx], gcv_values[optimum_idx], 'r*')
        # alpha = self.alphas[optimum_idx]
        # print("alpha: ", alpha)
        # plt.title(f"GCV: {alpha}")

        source_mat = self.inverse_operators[optimum_idx].data @ M
        return source_mat[0], optimum_idx
    
    
    def regularise_product(self, M):
        """ Find optimally regularized inverse solution using the product method [1].
        
        Parameters
        ----------
        M : numpy.ndarray
            The M/EEG data matrix (n_channels, n_timepoints)
        
        Return
        ------
        source_mat : numpy.ndarray
            The inverse solution  (dipoles x time points)
        optimum_idx : int
            The index of the selected (optimal) regularization parameter
        
        References
        ----------
        [1] Grech, R., Cassar, T., Muscat, J., Camilleri, K. P., Fabri, S. G.,
        Zervakis, M., ... & Vanrumste, B. (2008). Review on solving the inverse
        problem in EEG source analysis. Journal of neuroengineering and
        rehabilitation, 5(1), 1-33.

        """

        product_values = []

        for inverse_operator in self.inverse_operators:
            x = np.squeeze(inverse_operator.data @ M)

            M_hat = self.leadfield@x
            residual_norm = np.linalg.norm(M_hat - M)
            semi_norm = np.linalg.norm(x)
            product_value = semi_norm * residual_norm 
            product_values.append(product_value)

        optimum_idx = np.argmin(product_values)

        # plt.figure()
        # plt.plot(self.alphas, product_values)
        # plt.plot(self.alphas[optimum_idx], product_values[optimum_idx], 'r*')
        # alpha = self.alphas[optimum_idx]
        # plt.title(f"Product: {alpha}")

        source_mat = self.inverse_operators[optimum_idx].data @ M
        return source_mat[0], optimum_idx

    @staticmethod
    def delete_from_list(a, idc):
        ''' Delete elements of list at idc.'''

        idc = np.sort(idc)[::-1]
        for idx in idc:
            a.pop(idx)
        return a

    def find_corner(self, r_vals, l2_norms):
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

        A = np.array([r_vals[0], l2_norms[0]])
        C = np.array([r_vals[-1], l2_norms[-1]])
        areas = []
        for j in range(1, len(l2_norms)-1):
            B = np.array([r_vals[j], l2_norms[j]])
            AB = self.euclidean_distance(A, B)
            AC = self.euclidean_distance(A, C)
            CB = self.euclidean_distance(C, B)
            area = abs(self.calc_area_tri(AB, AC, CB))
            areas.append(area)
        if len(areas) > 0:
            idx = np.argmax(areas)+1
        else:
            idx = 0
        return idx

    @staticmethod
    def filter_norms(r_vals, l2_norms):
        ''' Filter l2_norms where they are not monotonically decreasing.

        Parameters
        ----------
        r_vals : [list, numpy.ndarray]
            List or array of r-values
        l2_norms : [list, numpy.ndarray]
            List or array of l2_norms
        
        Return
        ------
        bad_idc : list
            List where indices are increasing

        '''
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

    def prepare_forward(self):
        ''' Prepare leadfield for calculating inverse solutions by applying
        common average referencing and unit norm scaling.

        Parameters
        ----------
        

        Return
        ------
        '''
        # Check whether forward model has free source orientation
        # if yes -> convert to fixed
        if self.forward["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            print("Forward model has free source orientation. This is currently not possible, converting to fixed.")
            # convert to fixed
            self.forward = mne.convert_forward_solution(self.forward, force_fixed=True, verbose=0)
        
        self.leadfield = deepcopy(self.forward["sol"]["data"])
        
        if self.prep_leadfield:
            self.leadfield -= self.leadfield.mean(axis=0)
            self.leadfield /= np.linalg.norm(self.leadfield, axis=0)
            

    @staticmethod
    def euclidean_distance(A, B):
        ''' Euclidean Distance between two points (A -> B).'''
        return np.sqrt(np.sum((A-B)**2))

    @staticmethod
    def calc_area_tri(AB, AC, CB):
        ''' Calculates area of a triangle given the length of each side.'''
        s = (AB + AC + CB) / 2
        area = (s*(s-AB)*(s-AC)*(s-CB)) ** 0.5
        return area
        
    def source_to_object(self, source_mat):
        ''' Converts the source_mat matrix to the mne.SourceEstimate object.

        Parameters
        ----------
        source_mat : numpy.ndarray
            Source matrix (dipoles, time points)-

        Return
        ------
        stc : mne.SourceEstimate
        
        '''
        # Convert source to mne.SourceEstimate object
        source_model = self.forward['src']
        vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
        tmin = self.tmin
        sfreq = self.obj_info["sfreq"]
        tstep = 1/sfreq
        subject = self.obj_info["subject_info"]

        if type(subject) == dict and "his_id" in subject:
            subject = subject["his_id"]
        # else assume fsaverage as subject id
        else:
            subject = "fsaverage"
        
        stc = mne.SourceEstimate(source_mat, vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=self.verbose)
        return stc
    
    def save(self, path):
        ''' Saves the solver object. 

        Paramters
        ---------
        path : str
            The path to save the solver.
        Return
        ------
        self : BaseSolver
            Function returns itself.

        '''
        
    
        name = self.name

        # get list of folders in path
        list_of_folders = os.listdir(path)
        model_ints = []
        for folder in list_of_folders:
            full_path = os.path.join(path, folder)
            if not os.path.isdir(full_path):
                continue
            if folder.startswith(name):
                new_integer = int(folder.split('_')[-1])
                model_ints.append(new_integer)
        if len(model_ints) == 0:
            model_name = f'\\{name}_0'
        else:
            model_name = f'\\{name}_{max(model_ints)+1}'
        new_path = path+model_name
        os.mkdir(new_path)

        if hasattr(self, "model"):
            # Save model only
            self.model.save(new_path)

            # Save rest
            # Delete model since it is not serializable
            self.model = None
            self.generator = None
            with open(new_path + '\\instance.pkl', 'wb') as f:
                pkl.dump(self, f)
            
            # Attach model again now that everything is saved
            try:
                self.model = tf.keras.models.load_model(new_path, custom_objects={'loss': self.loss})
            except:
                print("Load model did not work using custom_objects. Now trying it without...")
                self.model = tf.keras.models.load_model(new_path)
        else:
            with open(new_path + '\\instance.pkl', 'wb') as f:
                pkl.dump(self, f)
        return self