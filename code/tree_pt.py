import numpy as np
from scipy.stats import multivariate_normal

import nrpt
import gaussian_tree
import var_pt

import matplotlib.pyplot as plt


## Tree PT implementation!
def tree_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = [(point, "g") for point in initial_state]
    
    d = 4
    reference = lambda x: multivariate_normal.logpdf(np.asarray(x), mean=np.zeros(d), cov=np.identity(d))
    
    Lambda_vs_r_points_TREE = []
    RestartRate_vs_r_points = []
    NumRestarts_vs_r_points = []
    
    toReturn = []
    for r in range(6, num_tuning_rounds):
        print("-----","TUNING ROUND", r,"-----")
        
        curr_state = [(tup[0], "g") for tup in curr_state]
        T = 2**r
        
        path = nrpt.path(schedule, reference, log_target)
        
        restarts, result = var_pt.vanilla_NRPT_with_RWMH(curr_state, schedule, path, T)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        for chain in samples: toReturn.append(chain[-1][0])
       
        temp_tree, global_nodes = gaussian_tree.tree_decomposition([chain[-1][0] for chain in samples])
        cl_tree = gaussian_tree.directed_graph(temp_tree)    
        reference = lambda x: gaussian_tree.tree_logpdf(cl_tree, x, global_nodes)
        
        schedule = var_pt.update_schedule(reject_rates, schedule)
        curr_state = samples[-1]
        
        print("Reject rates:",reject_rates)
        print("Schedule:",schedule)
        print()
        
        ## Collect points for Lambda vs. r plot
        Lambda_vs_r_points_TREE.append([r, np.sum(reject_rates / (1 - reject_rates))])
        
        ## Collect points for NumRestarts vs. r plot
        NumRestarts_vs_r_points.append([r, restarts])
        
        ## Collect points for Tau vs. r plot
        RestartRate_vs_r_points.append([r, restarts/T])
    

        
    return toReturn, reject_rates, Lambda_vs_r_points_TREE, NumRestarts_vs_r_points, RestartRate_vs_r_points

