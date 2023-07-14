
from diffusion_objective import DiffusionObjective
from jax import numpy as jnp
from dataset import Dataset
import torch as torch
import pandas as pd
import os
import numpy as np
from conHe_Param import conHe_Param
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint
from emcee_main import emcee_main
from generate_bounds import generate_bounds
import pickle
import random
import time
from plot_results import plot_results
from optimization_routines import diffEV_multiples
from save_results import save_results

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_input = pd.read_csv(f"{dir_path}/data/input_KM95-28-Dc-1250um.csv")
domains_to_model = 3
mineral_name = "quartz"
time_add = [3600*5,110073600]
temp_add = [40,21.111111111]
sample_name = "KM95-28-Dc"
moves = "snooker" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists
misfit_stat = "l1_moles"
omit_value_indices =  []



#################################################################

def organize_x(x,ndim):
        ndom = int(((ndim)/2))
        print(f"ndom is {ndom}")
        if len(x)%2 != 0:
            moles = x[0]
            x = x[1:]
        Ea = x[0]
        lnd0aa = x[1:1+ndom]
        fracs = x[1+ndom:]
        fracs = np.append(fracs,1-np.sum(fracs))
        
        n = len(fracs)
        # Traverse through all array elements
        for i in range(n):
            
            # Last i elements are already in place
            for j in range(0, n - i - 1):
                
                # Traverse the array from 0 to n-i-1
                # Swap if the element found is greater than the next element
                if lnd0aa[j] < lnd0aa[j + 1]:
                    lnd0aa[j], lnd0aa[j + 1] = lnd0aa[j + 1], lnd0aa[j]
                    fracs[j], fracs[j + 1] = fracs[j + 1], fracs[j]
        if len(x)%2 != 0:
            output = np.append(moles,Ea)
        else:
             output = Ea
        output = np.append(output,lnd0aa)
        output = np.append(output,fracs[0:-1])
        return output

# Create dataset class for each associate package

dataset = Dataset("diffEV", data_input)


objective = DiffusionObjective(
    "diffEV",
    dataset, 
    time_add = torch.tensor(time_add), 
    temp_add = torch.tensor(temp_add), 
    pickle_path = f"{dir_path}/data/lookup_table.pkl",
    omitValueIndices= omit_value_indices,
    stat = misfit_stat
)

# Read in the nonlinear constraint


params, misfit_val = diffEV_multiples(objective,dataset,1,mineral_name,domains_to_model)
start_time = time.time()


plot_results(params,dataset,objective,sample_name=sample_name )
save_results(domains_to_model,sample_name = sample_name,misfit_stat = misfit_stat)
print(organize_x(params,len(params)))

