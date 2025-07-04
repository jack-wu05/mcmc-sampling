import numpy as np
from scipy.stats import multivariate_normal

import nrpt
import gaussian_tree
import var_pt

import matplotlib.pyplot as plt


##### Tree PT implementation!
# See variational_PT_with_RWMH for comments
def tree_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target, target_mu, target_Sigma):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = [(point, "g") for point in initial_state]
    
    d = 4
    reference = lambda x: multivariate_normal.logpdf(np.asarray(x), mean=np.zeros(d), cov=np.identity(d))
    
    Lambda_vs_r = []
    RestartRate_vs_r = []
    NumRestarts_vs_r = []
    kl_vs_r = []
    restarts_vs_cost = []
    
    toReturn = []
    for r in range(6, num_tuning_rounds):
        print("-----","TUNING ROUND", r,"-----")
        
        curr_state = [(tup[0], "g") for tup in curr_state]
        T = 2**r
        cost = 0
        
        path = nrpt.path(schedule, reference, log_target)
        
        restarts, result = var_pt.vanilla_NRPT_with_RWMH(curr_state, schedule, path, T)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        for chain in samples: toReturn.append(chain[-1][0])
        
        
        ## Collect points for KL vs. r plot
        if r==6: 
            kl_vs_r.append([r, var_pt.kl_div(target_mu, target_Sigma, np.zeros(d), np.identity(d))])
        else:
            mu, dense = gaussian_tree.extract_tree_params(global_nodes, temp_tree)
            kl_vs_r.append([r, var_pt.kl_div(target_mu, target_Sigma, mu, dense)])
        
        
       
        temp_tree, global_nodes = gaussian_tree.tree_decomposition([chain[-1][0] for chain in samples])
        cl_tree = gaussian_tree.directed_graph(temp_tree)    
        reference = lambda x: gaussian_tree.tree_logpdf(cl_tree, x, global_nodes)
        
        schedule = var_pt.update_schedule(reject_rates, schedule)
        curr_state = samples[-1]
        
        cost += d**2 + T*num_chains*d
        
        print("Reject rates:",reject_rates)
        print("Schedule:",schedule)
        print()
        
        ## Collect points for Lambda vs. r plot
        Lambda_vs_r.append([r, np.sum(reject_rates / (1 - reject_rates))])
        
        ## Collect points for NumRestarts vs. r plot
        NumRestarts_vs_r.append([r, restarts])
        
        ## Collect points for Tau vs. r plot
        RestartRate_vs_r.append([r, restarts/T])
        
        ## Collect points for num restarts vs cost plot
        restarts_vs_cost.append([cost, restarts])
    

        
    return toReturn, reject_rates, Lambda_vs_r, NumRestarts_vs_r, RestartRate_vs_r, kl_vs_r, restarts_vs_cost

