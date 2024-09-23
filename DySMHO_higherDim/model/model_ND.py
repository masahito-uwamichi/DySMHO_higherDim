from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
from sklearn import linear_model

from scipy.signal import savgol_filter
import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import cvxopt

import re

from DySMHO_higherDim.model.utils_ND import time_scale_conversion, optim_solve, thresholding_accuracy_score, thresholding_mean_to_std, dyn_sim


class ND_MHL(): ###
    
    '''
    Initiating class with data:
        Inputs: 
        - y: N_t by N array of noisy state measurements (N_t is the number of measurements) ###
        - t: is the times at which the measurements in y were made (arbitrary units) ###
        - basis: list of length N with disctionaries corresponding to basis functions used for identification of the dynamics ###
    ''' 
    def __init__(self, y, t, basis):
        self.y = y
        self.t = t 
        self.basis = basis ###
        self.N = len(basis) ###

    '''
    Smoothing function applies the Savitzky-Golay filter to the state measurements
        Inputs:
        -smooth_iter: (interger) number of times the filter is repeatedly applied on the data
        -window_size: (interger) The length of the filter window (i.e., the number of coefficients
        -poly_order: (interger) The order of the polynomial used to fit the samples
    '''   
    def smooth(self, window_size = None, poly_order = 2, verbose = True): 
        
        if verbose: 
            print('\n')
            print('--------------------------- Smoothing data ---------------------------')
            print('\n')
            
        # Automatic tunning of the window size 
        if window_size == None: 

            for i in range(self.N):
                self.smooth_each_dim_without_window_size(i, poly_order = poly_order, verbose = verbose)

        # Pre-specified window size
        else: 
            for i in range(self.N):
                self.y[:,i] = savgol_filter(self.y[:,i], window_size, poly_order)
            
            self.t = self.t[:len(self.y)]
            
    def smooth_each_dim_without_window_size(self, i, poly_order = 2, verbose = True):
        y_norm = (self.y[:,i]-min(self.y[:,i]))/(max(self.y[:,i])-min(self.y[:,i]))     
        std_prev = np.std(diff(y_norm,1))
        window_size_used = 1 
        std = [] 
        while True:
            std.append(std_prev)
            window_size_used += 10 
            y_norm = savgol_filter(y_norm, window_size_used, poly_order)
            std_new = np.std(diff(y_norm,1))
            if verbose: 
                print('Prev STD: %.5f - New STD: %.5f - Percent change: %.5f' % (std_prev, std_new, 100*(std_new-std_prev)/std_prev))
            if abs((std_new-std_prev)/std_prev)  < 0.1: 
                window_size_used -= 10
                break   
            else:
                std_prev = std_new
                y_norm = (self.y[:,i]-min(self.y[:,i]))/(max(self.y[:,i])-min(self.y[:,i])) 
                        
        if window_size_used > 1: 
            print('Smoothing window size (dimension '+str(i+1)+'): '+str(window_size_used),'\n')
            self.y[:,i] = savgol_filter(self.y[:,i], window_size_used, poly_order)
        else: 
            print('No smoothing applied')
            print('\n')


    '''
    First pre-processing step which includes Granger causality analysis for derivative and basis functions 
        Inputs: 
        - granger: (boolean) whether Granger causality test is performed to filter the original basis or not 
        - significance: (real, lb = 0, ub = 1) significance level for p-values obatined ivia Granger causality test 
    '''
    def pre_processing_1(self, 
                         granger = True, 
                         significance = 0.1,
                         verbose = True,
                         rm_features = None): ### 
        
        if rm_features == None:
            rm_features = [[]] * self.N ###
        
        # Computing derivatives using finite differences 
        dy_dt = []

        for i in range(self.N):
            dy_dt.append((self.y[2:,i] - self.y[0:-2,i])/(self.t[2:] - self.t[:-2]))
        dydt = np.column_stack(dy_dt)
        self.t_diff = self.t[:-1]

        self.df_y = []
        self.dy_dt = []
        self.all_features = []
       
        self.columns_to_keep = [[]] * self.N

        for i in range(self.N):
            self.pre_processing_1_each_dim(i, dydt, granger, significance, verbose, rm_features[i])

        
        
    def pre_processing_1_each_dim(self, i, dydt, granger, significance, verbose, rm_features):
        # Generating features in pandas dataframe 
        df_y = pd.DataFrame() 
        for k, basis_fun_i in enumerate(self.basis[i]['functions']): 
            df_y[self.basis[i]['names'][k]] = [basis_fun_i(j[0],j[1],j[2],j[3]) for j in self.y[1:-1]]

        df_y['dy_dt'] = dydt[:,i]
        df_y.drop(df_y.tail(1).index,inplace=True)
        df_y['y_shift'] = self.y[2:-1,i]

        self.df_y.append(df_y)
        self.dy_dt.append(dydt[:,i])
        self.all_features.append(df_y.columns)


        if '1' in self.df_y[i].columns: 
            self.columns_to_keep[i] = ['1']
        else: 
            self.columns_to_keep[i] = []

        self.dy_dt[i] = df_y['dy_dt']

        if granger:
            tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']

            granger_causality = {}
            for j in df_y.columns: 
                if j != 'dy_dt': 
                    x = df_y[j].dropna()
                    y = df_y['y_shift']
                    data = pd.DataFrame(data = [y,x]).transpose()
                    x = grangercausalitytests(data, 1, addconst=True, verbose=False)
                    p_vals = [x[1][0][test][1] for test in tests]

                    granger_causality[j] = [np.mean(p_vals), np.std(p_vals)]

            df = pd.DataFrame.from_dict(granger_causality).T
            count = 0
            for j in df.index:
                if df[0][j] < significance and j != 'dy_dt': 
                    self.columns_to_keep[i].append(j)
                count += 1

            if verbose:
                print('--------- Pre-processing 1: Dimension %i ---------'% (i+1))
                print(df,'\n')
                print('Columns to keep for y%i: '%(i+1), self.columns_to_keep[i])
                print('\n')

        self.df_y[i].drop([j for j in self.df_y[i].columns if j not in self.columns_to_keep[i]], axis = 1, inplace = True )

        for j in rm_features: 
            if j in self.columns_to_keep[i]: 
                self.columns_to_keep[i].remove(j)



    '''
    Second pre-processing step which includes Ordinary Least Squares (OLS) for derivative and basis functions 
        Inputs: 
        - intercept: (boolean) wether a constant term is added to the regression problem (i.e., basis function of 1 )
        - verbose: print outputs of OLS 
        - plot: plot derivatives and resulting fit 
        - significance: (real, lb = 0, ub = 1) significance level for p-values obatined via OLS to determine non-zero coefficients 
        - confidence: (real, lb = 0, ub = 1) confidence level used to derive bounds for the non-zero parameters identified in OLS 
    '''
    def pre_processing_2(self, intercept = None, verbose = True, plot = False, significance = 0.9, confidence = 1-1e-8 ):  ###

        if intercept == None:
            intercept = [True] * self.N

        self.initial_theta = [] 
        self.theta_bounds = []
        self.non_zero = [] 
        self.p_val_tolerance = significance
        self.confidence_interval = 1 - confidence
        self.all_features_sym = [] 

        for i in range(self.N):
            self.pre_processing_2_each_dim(i, intercept[i], verbose, plot)

        
    def pre_processing_2_each_dim(self, i, intercept, verbose, plot): 
        
        X_train = self.df_y[i].to_numpy() 
        y_train = self.dy_dt[i].to_numpy()
        model = sm.OLS(y_train,X_train)
        results1 = model.fit()
        if verbose: 
            print('\n')
            print('--------- Pre-processing 2: Dimension 1 ---------\n')
            print(results1.summary()) 
            
        if plot: 
            prstd, iv_l, iv_u = wls_prediction_std(results1)
            plt.figure() 
            plt.plot(y_train, color = '#d73027', linewidth = 3)
            gray = [102/255, 102/255, 102/255]
            plt.plot(np.dot(X_train, results1.params), color = 'k', linewidth = 3)
            plt.legend(['Derivative data','Model prediction'])
            plt.title('OLS $y_1$')
            plt.show()

        conf_interval1 = results1.conf_int(alpha = self.confidence_interval) 
        count = 0
        count_vars = i
        
        for j in self.all_features[i]: 
            if j not in ['dy_dt','y_shift']: 
                self.all_features_sym.append(j)
                if (j in self.columns_to_keep1):
                    if (results1.pvalues[count]) < (self.p_val_tolerance or j == '1') or bool(re.match(r'^y\d+$', j)):  ###
                        self.initial_theta.append(results1.params[count])
                        self.theta_bounds.append((conf_interval1[count][0],conf_interval1[count][1]))
                        self.non_zero.append(count_vars)
                    else: 
                        self.initial_theta.append(0)
                        self.theta_bounds.append((0,0))
                    count += 1


                elif (j not in self.columns_to_keep1): 
                    self.initial_theta.append(0)
                    self.theta_bounds.append((0,0))

        
    '''
    Performs moving horizon dicovery routine
        Inputs: 
        -
        
    '''
    def discover(self, 
                 horizon_length, 
                 time_steps, data_step, 
                 optim_options = {'nfe':50, 'ncp':15}, 
                 thresholding_frequency = 20, 
                 thresholding_tolerance = 1,
                 sign = False): 
        
        y_init = self.y[0,:]
        y0_step = self.y[0:len(self.y) + 1:data_step]
        
        # Initializing iterations and error
        iter_num = 0
        thresholded_indices = [i for i,j in enumerate(self.initial_theta) if i not in self.non_zero ]
        len_thresholded_indices_prev = [len(thresholded_indices), 0]
        theta_init_dict = {i:j for i,j in enumerate(self.initial_theta)}
        error = []
        theta_updates = {0: self.initial_theta} 
        self.number_of_terms = [len(thresholded_indices)]
        # Parameter values after each OLS step
        self.theta_after_OLS = [self.initial_theta]
        self.CV = [] 
        
        for k, t in enumerate(self.t[0:len(self.t) - 1:data_step]):

            if t + horizon_length < self.t[-1]:
                
                # Obtaining collocation time scale for current step
                y, t_col = time_scale_conversion(t, 
                                                 horizon_length, 
                                                 optim_options, 
                                                 self.t, 
                                                 self.y)

                
                # Performing optimization to compute the next theta
                theta_init, error_sq = optim_solve(y_init, 
                                                   [t, t + horizon_length], 
                                                   theta_init_dict, 
                                                   self.theta_bounds, 
                                                   y, 
                                                   self.basis, ###
                                                   self.all_features_sym, 
                                                   iter_num, 
                                                   thresholded_indices, 
                                                   optim_options,
                                                   sign)
                error.append(error_sq)
        

                # Updating theta
                theta_updates[iter_num] = theta_init
                theta_init_dict = {i:j for i,j in enumerate(theta_init)} 
                
            
                
                # Determining parameters to threshold
                thresholded_indices, CV = thresholding_mean_to_std(len(self.initial_theta), 
                                                               thresholded_indices, 
                                                               theta_updates, 
                                                               iter_num, 
                                                               self.t, 
                                                               self.y,
                                                               iter_thresh = thresholding_frequency, 
                                                               tolerance = thresholding_tolerance)
                self.number_of_terms.append(len(thresholded_indices))
                if len(CV) > 0: 
                    self.CV.append(CV)
                    
                
                # Beaking loop is the thresholded parametrs have not changed in 3 rounds of thresholding
                print('\n')
                if len(thresholded_indices) == len_thresholded_indices_prev[0]:
                    if len_thresholded_indices_prev[1] < 4*thresholding_frequency: 
                        len_thresholded_indices_prev[1] += 1 
                    else: 
                        break
                else: 
                    len_thresholded_indices_prev[0] = len(thresholded_indices) 
                    len_thresholded_indices_prev[1] = 0 
                    
                len_basis = np.array([0]+[len(bas) for bas in self.basis])
                len_basis = np.cumsum(len_basis)

                # Recomputing bounds once some of the parameters have been eliminated
                if not iter_num % thresholding_frequency and iter_num > 0:

                    # Dropping columns in the dataframe containing the evaluated basis functions 
                    for k in range(self.N):
                        self.df_y[k].drop([j for i,j in enumerate(self.all_features_sym) if i in thresholded_indices and j in self.df_y[k].columns], axis = 1, inplace = True )
                        self.columns_to_keep[k] = self.df_y[k].columns                    
                    
                    # Running pre-processing again (OLS) -- to obatin better bounds for the parameters that remain 
                    self.pre_processing_2(verbose = True, 
                                          plot = False, 
                                          significance = 0.7, 
                                          confidence = 1-1e-8 )
                    thresholded_indices = [i for i,j in enumerate(self.initial_theta) if i not in self.non_zero ]
                    theta_init_dict = {i:j for i,j in enumerate(self.initial_theta)}
                    theta_updates[iter_num] = self.initial_theta
                    theta_init_dict = {i:j for i,j in enumerate(self.initial_theta)}
                    self.theta_after_OLS.append(self.initial_theta)

                # Obtaining the next initial condition
                if k + 1 < len(self.y):
                    y_init = [y0_step[k + 1, 0], y0_step[k + 1, 1], y0_step[k + 1, 2], y0_step[k + 1, 3]] ###

                iter_num += 1
                
        self.theta_values = theta_updates
        
        
    def validate(self, xs_validate, y_validate, metric = 'MSE', plot = True): 
        
        theta_values = pd.DataFrame(self.theta_values)
        theta_values.loc[theta_values.iloc[:,-1] == 0, :] = 0
        mean_theta = theta_values.iloc[:,-30:-1].mean(axis=1).to_numpy()
        
        ys_mhl = dyn_sim(mean_theta, 
                         xs_validate,
                         y_validate,
                         self.basis) ###
        
        self.y_simulated = ys_mhl
        
        if metric == 'MSE':
            from sklearn.metrics import mean_squared_error
            error = 0
            for i in range(self.N):
                error += mean_squared_error(y_validate[:,i],ys_mhl[:,i])
            print('\n', 'MSE: %.10f '% error)
            
        self.error = error 
        

        if plot == True:
            for i in range(self.N):
                plt.plot(xs_validate, y_validate[:, i], 'o')
                plt.plot(xs_validate, ys_mhl[:, i], color='black')
            plt.show()
                
        
                
                
                
                
            
            

        
        
    
         