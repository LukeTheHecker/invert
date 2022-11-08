from copy import deepcopy
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
    '''
    Parameters
    ----------
    regularisation_method : str
        Can be either 
            "GCV"       -> generalized cross validation
            "L"         -> L-Curve method using triangle method
            "Product"   -> Minimal product method

    n_reg_params : int
        The number of regularisation parameters to use. The higher, the 
        more accurate the regularisation and the slower the computations.
    car_leadfield : bool
        If True -> Apply common average referencing on the leadfield columns.
    '''
    def __init__(self, regularisation_method="GCV", n_reg_params=24, 
        car_leadfield=True, verbose=0):
        self.verbose = verbose
        # self.r_values = np.insert(np.logspace(-10, 10, n_reg_params), 0, 0)
        self.r_values = np.insert(np.logspace(-5, 2, n_reg_params), 0, 0)
        # self.r_values = np.arange(13)
        
        # self.alphas = deepcopy(self.r_values)
        self.regularisation_method = regularisation_method
        self.car_leadfield = car_leadfield
        
    def make_inverse_operator(self, forward: mne.Forward, *args, alpha="auto", **kwargs):
      
        self.forward = forward
        self.prepare_forward()
        self.leadfield = self.forward['sol']['data']
        self.alpha = alpha
        self.alphas = self.get_alphas()

        pass

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        ''' Apply the inverse operator '''
        evoked = self.prep_data(evoked)

        if len(self.inverse_operators) == 1:
            source_mat = self.inverse_operators[0].apply(evoked)
        else:
            if self.regularisation_method.lower() == "l":
                source_mat = self.regularise_lcurve(evoked)
            elif self.regularisation_method.lower() == "gcv":
                source_mat = self.regularise_gcv(evoked)
            elif self.regularisation_method.lower() == "product":
                source_mat = self.regularise_product(evoked)
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
        # print("L-CURVE")
        M = evoked.data
        leadfield = self.forward["sol"]["data"]
        source_mats = [inverse_operator.apply( evoked ) for inverse_operator in self.inverse_operators]
        
        M -= M.mean(axis=0)
        leadfield -= leadfield.mean(axis=0)
        

        l2_norms = [np.log(np.linalg.norm( leadfield @ source_mat )) for source_mat in source_mats]
        residual_norms = [np.log(np.linalg.norm( leadfield @ source_mat - M )) for source_mat in source_mats]


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

        return source_mat
        
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
        # print("GCV")
        self.leadfield = self.forward["sol"]["data"]
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

        optimum_idx = np.argmin(gcv_values[1:])+1

        # plt.figure()
        # plt.loglog(self.alphas, gcv_values, 'ok')
        # plt.plot(self.alphas[optimum_idx], gcv_values[optimum_idx], 'r*')
        # alpha = self.alphas[optimum_idx]
        # print("alpha: ", alpha)
        # plt.title(f"GCV: {alpha}")

        source_mat = self.inverse_operators[optimum_idx].data @ M
        return source_mat[0]
    
    
    def regularise_product(self, evoked):
        # print("Product")
        self.leadfield = self.forward["sol"]["data"]
        M = evoked.data
        product_values = []

        for alpha, inverse_operator in zip(self.alphas, self.inverse_operators):
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
        return source_mat[0]

    @staticmethod
    def delete_from_list(a, idc):
        ''' Delete elements of list at idc.'''

        idc = np.sort(idc)[::-1]
        for idx in idc:
            a.pop(idx)
        return a

    def find_corner(self, l2_norms, residual_norms):
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

        A = np.array([residual_norms[0], l2_norms[0]])
        C = np.array([residual_norms[-1], l2_norms[-1]])
        areas = []
        for j in range(1, len(l2_norms)-1):
            B = np.array([residual_norms[j], l2_norms[j]])
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
        if self.car_leadfield:
            self.forward["sol"]["data"] -= self.forward["sol"]["data"].mean(axis=0)

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

        if type(subject) == dict:
            subject = "bst_raw"

        if subject is None:
            subject = "fsaverage"
        
        stc = mne.SourceEstimate(source_mat, vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=self.verbose)
        return stc