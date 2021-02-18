# Libraries
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.stats


def randomize(tolerance, dist="Nominal", cp = 1.67):
    """
    Function that returns a value within specified margins given component properties.
    Input:
       - Tolerance: Numpy Array. Properties of the electronic component. Array with shape = (3,).
       - dist: String. Statistical distribution from which to sample.
         * Uni: samples a value from a uniform distribution.
         * Gauss: samples a value from a Gaussian distribution.
         * Nominal: returns the nominal value.
       - cp: Float. Process capability (6 sigma context) required for the component.
    Output:
       - Float. Sampled from distribution.
    """
    if dist == "Uni":
        return random.uniform(tolerance[0], tolerance[2])
    elif dist == "Gauss":
        mu = tolerance[1]
        sigma = (tolerance[2]-tolerance[0])/(6 * cp)
        return random.gauss(mu, sigma)
    else:
        return tolerance[1]
    

def standarize(y,pct,pct_lower):
    sc = StandardScaler() 
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y):int(len_y * pct)]
    len_y = len(y)
    yy=([[x] for x in y])
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    return y_std,len_y,y
    

def simulation(n_points, base_model, base_class, dist):
    """
    Function that simulates observations for bla bla. 
    Input:
    - n_points : number of points simulated
    - dist_components: distributon assumed for the components
    """
    y = np.zeros(n_points)
    for i in range(n_points):
        
        # Randomize numbers
        Vout_r = randomize(base_model.Vout, dist)
        LS_Ron_r = randomize(base_model.LS_Ron, dist)
        Iout_r = randomize(base_model.Iout, dist)
        Vin_r = randomize(base_model.Vin, dist)
        Fsw_r = randomize(base_model.Fsw, dist)
        Vbody_diode_r = randomize(base_model.Vbody_diode, dist)
        L_r = randomize(base_model.L, dist)
        DCR_r = randomize(base_model.DCR, dist)
        P_IC_r = randomize(base_model.P_IC, dist)
        
        #Sensible Analysis
        # modify one value to plug into the class
        # "sens_var" = randomize(base_model."sens_var", sens_dist)

        sim_PSU = base_class(Vout_r, LS_Ron_r, Iout_r, Vin_r, Fsw_r, Vbody_diode_r, L_r, DCR_r, P_IC_r)
        
        y[i] = sim_PSU.P_in()
    return y

# github.com/samread81/Distribution-Fitting-Used_Car_Dataset/blob/master/Workbook.ipynb
def compute_chi_square(data):
    y,size,_ = standarize(data, 0.99, 0.01)
    dist_names = ['weibull_min', 'norm', 'weibull_max', 'beta', 'invgauss',
                  'uniform', 'gamma', 'expon', 'lognorm', 'pearson3', 'triang']
    
    chi_square_statistics = []
    log_likelihood = []
    parameters = []
    
    # Bins
    percentile_bins = np.linspace(0,100, 11)
    percentile_cutoffs = np.percentile(y, percentile_bins)
    obs_frequency, _ = np.histogram(y, bins=percentile_cutoffs)
    cum_obs_frequency = np.cumsum(obs_frequency)
    
    # Check candidate distributions
    for distribution in dist_names:
        # Set up candidate distribution
        
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y)
        parameters.append(param)
        log_likelihood.append(np.sum(np.log(dist.pdf(y, *param))))
        print('Distribution: ' + distribution + ' || Parameters: ' + str(param))# + '|| Log-likelihood: ' + 
              #str(log_likelihood) + '\n')
        
        # CDF
        cdf_fit = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fit[bin+1] - cdf_fit[bin]
            expected_frequency.append(expected_cdf_area)
        
        # Chi-Square
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = round(sum (((cum_expected_frequency - cum_obs_frequency) ** 2) / cum_obs_frequency), 0)
        chi_square_statistics.append(ss)
    
    # Sort
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['Log_likelihood'] = log_likelihood
    results['Chi_square'] = chi_square_statistics
    results['Parameters'] = parameters
    results.sort_values(['Chi_square'], inplace=True)
    
    print('\nDistributions listed by Goodness of Fit:')
    print('..........................................')
    print(results)  
    
    return results

def LRT(best_options, n_datasets, n_sim):
    dist_A = best_options.iloc[0]['Distribution']
    dist_B = best_options.iloc[1]['Distribution']
    ll_A = best_options.iloc[0]['Log_likelihood']
    ll_B = best_options.iloc[1]['Log_likelihood']
    param = best_options.iloc[0]['Parameters']
    
    Q = 2 * (ll_B - ll_A)
    Q_array = np.zeros(n_datasets)
    
    for i in range(n_datasets):
        # Generate dataset
        sampling_dist = getattr(scipy.stats, dist_A)
        dataset = sampling_dist.rvs(*param, size = n_sim)
        
        # Fit models
        dist_A_i = getattr(scipy.stats, dist_A)
        param_A_i = dist_A_i.fit(dataset)
        ll_A_i = np.sum(np.log(dist_A_i.pdf(dataset, *param_A_i)))

        dist_B_i = getattr(scipy.stats, dist_B)
        param_B_i = dist_B_i.fit(dataset)
        ll_B_i = np.sum(np.log(dist_B_i.pdf(dataset, *param_B_i)))
        
        # Compute Qi
        Q_i = 2 * (ll_B_i - ll_A_i)
        Q_array[i] = Q_i 
    
    Quantile_Q = np.quantile(Q_array, 0.95)

    return Q, Quantile_Q