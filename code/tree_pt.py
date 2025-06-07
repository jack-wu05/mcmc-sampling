import numpy as np
import warnings
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator
import pints
import pints.toy
from scipy.stats import multivariate_normal

import nrpt
import gaussian_tree
import var_pt

import matplotlib.pyplot as plt

s = 2

def RWMH_exploration_kernel(log_gamma, initial_x, num_iters):
    curr_point = initial_x
    d = 4
    
    samples = np.zeros((num_iters, d))
    print(samples)
    
    for i in range(num_iters):
        new_point = np.zeros(d)
        
        for j in range(d):
            new_point[j] = np.random.normal(curr_point[j], s, 1)
            
        p = min(1, np.exp(log_gamma(new_point)-log_gamma(curr_point)))
    
        if np.random.rand() < p:
            curr_point = new_point
    
        samples[i] = curr_point
        
    return samples


def vanilla_NRPT_with_RWMH(initial_state, betas, log_annealing_path, num_iterations):
    num_distributions = len(betas)
    d = 4

    samples = []
    reject_rates = np.zeros(num_distributions-1)
    
    x_at_tminus1 = initial_state.copy()
    x_at_t = np.zeros((num_distributions, d))

    for t in range(num_iterations):
        for init in range(num_distributions):
            log_gamma = log_annealing_path[init]

            x_at_t[init] = RWMH_exploration_kernel(log_gamma, x_at_tminus1[init], 15)[-1]
        
        temp_alpha_vector = np.zeros(num_distributions)

        for i in range(num_distributions-1):
            temp_alpha_vector[i] = nrpt.alpha(log_annealing_path[i], log_annealing_path[i+1], x_at_t[i], x_at_t[i+1])
            reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i]) / 100
            
            if ((i%2==0 and t%2==0) or (i%2==1 and t%2==1)) and (np.random.rand() <= temp_alpha_vector[i]):
                x_at_t[i], x_at_t[i+1] = x_at_t[i+1], x_at_t[i]

        x_at_tminus1 = x_at_t.copy()
        samples.append(x_at_tminus1.copy())

    return {
        "reject_rates": reject_rates,
        "samples": samples
    }
    
    
## Variational PT implementation!
def variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = initial_state
    
    d = 4
    curr_reference = lambda x: multivariate_normal.pdf(x, mean=np.zeros(d), cov=np.eye(d))
    
    toReturn = []
    for r in range(1, num_tuning_rounds):
        T = 2**r
        
        path = nrpt.path(schedule, curr_reference, log_target)
        
        result = vanilla_NRPT_with_RWMH(curr_state, schedule, path, T)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        if r >= 5: 
            for chain in samples: toReturn.append(chain[-1])
        
        schedule = var_pt.update_schedule(reject_rates, schedule)
        
        if r >= 5:
            cl_tree = gaussian_tree.directed_graph(
                gaussian_tree.tree_decomposition(np.array([chain[-1] for chain in samples]))
                )    
            curr_reference = lambda x: gaussian_tree.tree_pdf(cl_tree, x)
        
        curr_state = samples[-1]
    
    return toReturn, reject_rates


## Kolmogorov-Smirnov test for correctness of Chow-Liu
# log_gamma = lambda x: -x**2 / 2
# num_samples = 3000
# iid_samples = np.random.normal(size=num_samples)

# kernel_samples = np.zeros(num_samples)
# for i in range(num_samples):
#     kernel_samples[i] = RWMH_exploration_kernel(log_gamma, iid_samples[i], 4000)[-1]

# ks_result = ks_2samp(iid_samples, kernel_samples)
# print("Kernel test p-value:", ks_result.pvalue)


### Toy example

d = 4
log_target = lambda x: multivariate_normal.logpdf(np.array(x), mean=np.zeros(d), cov=np.eye(d))


num_chains = 15
initial_state = [[0.25, 0.25, 0.25, 0.25]] * num_chains
num_tuning_rounds = 12

samples, rates = variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target)
samples = np.array(samples)
print("Mean vector:", np.mean(samples, axis=0))
print("Variance vector:", np.var(samples, axis=0))
        
        
        
        
        
    