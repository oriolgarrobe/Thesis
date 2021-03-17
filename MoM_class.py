import numpy as np
import math
import scipy.stats

class Method_of_Moments:
    def __init__(self, data):
        self.data = data
        
    def get_moments(self, data):
        moments={}
        moments['mean'] = np.mean(data)
        moments['variance'] = np.var(data)
        moments['skew'] = scipy.stats.skew(data)
        moments['kurtosis'] = scipy.stats.kurtosis(data)
        return moments
    
    def uniform_from_moments(self):
        return min(self.data), max(self.data) - min(self.data)
    
    # Reference Wackerly [Book Stat Methods]
    def beta_from_moments(self):
        # Scale data
        beta_data = (self.data - min(self.data)) / (max(self.data) - min(self.data))

        # Moments of data
        moments = self.get_moments(beta_data)

        # Sample Measures
        Y_bar = moments['mean']
        s2 = moments['variance']

        # Distribution Parameters
        alpha = Y_bar * (Y_bar*(1 - Y_bar)/s2 - 1)
        beta = alpha * (1 - Y_bar) / Y_bar
        loc = min(self.data)
        scale = max(self.data) - min(self.data)

        return alpha, beta, loc, scale
    
    def gamma_from_moments(self):
        # Scale data
        gamma_data = self.data - min(self.data)

        # Moments of data
        moments = self.get_moments(gamma_data)

        # Sample Measures
        Y_bar = moments['mean'] 
        n = len(self.data)

        # Distribution Parameters
        alpha = Y_bar**2 / moments['variance']
        beta = Y_bar / alpha
        loc = min(self.data)

        return alpha, loc, beta
    
    def norm_from_moments(self):
        # Moments of data
        moments = self.get_moments(self.data)

        # Distribution Parameters
        mu = moments['mean']
        sigma = moments['variance']**0.5

        return mu, sigma
    
    def lognorm_from_moments(self):
        #Scale Data
        lognorm_data = self.data - min(self.data)

        # Moments of data
        moments = self.get_moments(lognorm_data)

        # Sample Measures
        Y_bar = moments['mean']
        s2 = moments['variance']

        # Distribution parameters
        mu = math.log(Y_bar) - 0.5*math.log(s2/Y_bar + 1)
        sigma2 = math.log(s2/Y_bar + 1)

        return sigma2**0.5, min(self.data), math.exp(mu)