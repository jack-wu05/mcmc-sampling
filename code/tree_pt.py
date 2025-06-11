import numpy as np
from scipy.stats import multivariate_normal

import nrpt
import gaussian_tree
import var_pt

import matplotlib.pyplot as plt


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

            x_at_t[init] = var_pt.RWMH_exploration_kernel(log_gamma, x_at_tminus1[init], 60)[-1]
        
        temp_alpha_vector = np.zeros(num_distributions)

        for i in range(num_distributions-1):
            temp_alpha_vector[i] = nrpt.alpha(log_annealing_path[i], log_annealing_path[i+1], x_at_t[i], x_at_t[i+1])
            reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i]) / num_iterations
            
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
    reference = lambda x: multivariate_normal.logpdf(np.asarray(x), mean=np.zeros(d), cov=np.identity(d))
    
    Lambda_vs_r_points_TREE = []
    
    toReturn = []
    for r in range(6, num_tuning_rounds):
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
        print()
        
        ## Collect points for Lambda vs. r plot
        Lambda_vs_r_points_TREE.append([r, np.sum(reject_rates / (1 - reject_rates))])
    
    return toReturn, reject_rates, Lambda_vs_r_points_TREE

