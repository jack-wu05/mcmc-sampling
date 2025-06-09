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

s = 0.7

def RWMH_exploration_kernel(log_gamma, initial_x, num_iters):
    curr_point = initial_x
    d = 4
    samples = np.zeros((num_iters, d))
    
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
            reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i]) / 1000
            
            if ((i%2==0 and t%2==0) or (i%2==1 and t%2==1)) and (np.random.rand() <= temp_alpha_vector[i]):
                x_at_t[i], x_at_t[i+1] = x_at_t[i+1], x_at_t[i]

        x_at_tminus1 = x_at_t.copy()
        samples.append(x_at_tminus1.copy())

    return {
        "reject_rates": reject_rates,
        "samples": samples
    }
    
    
## Tree PT implementation!
def tree_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = initial_state
    
    d = 4
    reference = lambda x: multivariate_normal.logpdf(np.array(x), mean=np.zeros(d), cov=np.identity(d))
    
    Lambda_vs_r_points_TREE = []
    
    toReturn = []
    for r in range(4, num_tuning_rounds):
        print("-----","TUNING ROUND", r,"-----")
        
        T = 2**r
        
        path = nrpt.path(schedule, reference, log_target)
        
        result = vanilla_NRPT_with_RWMH(curr_state, schedule, path, T)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        for chain in samples: toReturn.append(chain[-1])
       
        cl_tree = gaussian_tree.directed_graph(
                gaussian_tree.tree_decomposition([chain[-1] for chain in samples])
            )    
        reference = lambda x: gaussian_tree.tree_pdf(cl_tree, x)
        
        schedule = var_pt.update_schedule(reject_rates, schedule)
        curr_state = samples[-1]
        
        print("Reject rates:",reject_rates)
        print("Schedule:",schedule)
        
        ## Collect points for Lambda vs. r plot
        Lambda_vs_r_points_TREE.append([r, np.sum(reject_rates / (1 - reject_rates))])
    
    return toReturn, reject_rates, Lambda_vs_r_points_TREE


def plot(points1, points2):
    x1,y1 = zip(*points1)  
    x2,y2 = zip(*points2)
      
    plt.figure()
    plt.plot(x1,y1, marker='o', color='b')
    plt.plot(x2,y2, marker = 'o', color='r')
    plt.grid(True)
    plt.show()
    
    
    
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
mean = [1,2,3,4]
cov = np.array([
    [1.0,  0.8,  0.5,  0.3],
    [0.8,  1.5,  0.6,  0.4],
    [0.5,  0.6,  2.0,  0.7],
    [0.3,  0.4,  0.7,  1.8]
])
log_target = lambda x: multivariate_normal.logpdf(np.array(x), mean=mean, cov=cov)

samples = []
for i in range(100000):
    sample = np.random.multivariate_normal(mean=mean, cov=cov)
    samples.append(sample)
print("True mean:", np.mean(samples, axis=0))
print("True variance:", np.var(samples, axis=0))
print()



num_chains = 15
initial_state = [[0.25, 0.25, 0.25, 0.25]] * num_chains
num_tuning_rounds = 14
samples, rates, Lambda_vs_r_points_TREE = tree_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target)
samples = np.array(samples)
print("Estimated mean:", np.mean(samples, axis=0))
print("Estimated variance:", np.var(samples, axis=0))

plot(Lambda_vs_r_points_TREE, Lambda_vs_r_points_VAR)







        
        
    