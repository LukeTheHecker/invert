from copy import deepcopy
import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import pickle as pkl
import tensorflow as tf

    
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        ''' Apply the inverse operator
        
        Parameters
        ----------
        evoked : mne.Evoked
            The Evoked data object
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne SourceEstimate object.
        
        '''
        evoked = self.prep_data(evoked)

        if len(self.inverse_operators) == 1:
            source_mat = self.inverse_operators[0].apply(evoked)
        elif self.use_last_alpha and self.last_reg_idx is not None:
            source_mat = self.inverse_operators[self.last_reg_idx].apply( evoked ) 
            
        else:
            if self.regularisation_method.lower() == "l":
                source_mat, idx = self.regularise_lcurve(evoked)
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() == "gcv":
                source_mat, idx = self.regularise_gcv(evoked)
                self.last_reg_idx = idx
            elif self.regularisation_method.lower() == "product":
                source_mat, idx = self.regularise_product(evoked)
                self.last_reg_idx = idx
            else:
                msg = f"{self.regularisation_method} is no valid regularisation method."
                raise AttributeError(msg)
            
            
        # print(type(source_mat), source_mat.shape)
        stc = self.source_to_object(source_mat, evoked)
        return stc
        
    @staticmethod
    def prep_data(evoked):
        if not evoked.proj:
            evoked.set_eeg_reference("average", projection=True, verbose=0).apply_proj(verbose=0)
        
        return evoked

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
        max_eig = eigs.max()
        if self.alpha == "auto":
            
            alphas = list(max_eig * self.r_values)
        else:
            alphas = [self.alpha*max_eig, ]
        return alphas

    def regularise_lcurve(self, evoked):
        """ Find optimally regularized inverse solution using the L-Curve method [1].
        
        Parameters
        ----------
        evoked : mne.Evoked
            The mne Evoked object
        
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
        M = evoked.data
        leadfield = self.leadfield
        source_mats = [inverse_operator.apply( evoked ) for inverse_operator in self.inverse_operators]
        
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

    def regularise_gcv(self, evoked):
        """ Find optimally regularized inverse solution using the generalized
        cross-validation method [1].
        
        Parameters
        ----------
        evoked : mne.Evoked
            The mne Evoked object
        
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
        M = evoked.data
        # M -= M.mean(axis=0)
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
        
        # Filter gcv_values that first increase
        try:
            keep_idx = np.where((np.diff(gcv_values)<0))[0][0]
        except:
            keep_idx = 0

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
    
    
    def regularise_product(self, evoked):
        """ Find optimally regularized inverse solution using the product method [1].
        
        Parameters
        ----------
        evoked : mne.Evoked
            The mne Evoked object
        
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

        M = evoked.data
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
        
    def source_to_object(self, source_mat, evoked):
        ''' Converts the source_mat matrix to the mne.SourceEstimate object.

        Parameters
        ----------
        source_mat : numpy.ndarray
            Source matrix (dipoles, time points)-
        evoekd : mne.Evoked
            Evoked data object.

        Return
        ------
        stc : mne.SourceEstimate
        
        '''
        # Convert source to mne.SourceEstimate object
        source_model = self.forward['src']
        vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
        tmin = evoked.tmin
        sfreq = evoked.info["sfreq"]
        tstep = 1/sfreq
        subject = evoked.info["subject_info"]

        if type(subject) == dict:
            subject = "bst_raw"

        if subject is None:
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