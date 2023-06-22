
from diffusion_objective import DiffusionObjective
from diffusion_problem import DiffusionProblem
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

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_input = pd.read_csv(f"{dir_path}/data/input_n13ksp_moles.csv")
domains_to_model = 6
mineral_name = "kspar"
time_add = [0,0]
temp_add = [0,0]
sample_name = "n13ksp_moles"
moves = "snooker" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists
misfit_stat = "l2_frac"
omit_value_indices =  [32,33,34,35,36,37,38,39,40,41]



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
        output = np.append(moles,Ea)
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


params, misfit_val = diffEV_multiples(objective,dataset,10,mineral_name,domains_to_model)
start_time = time.time()


plot_results(params,dataset,objective,sample_name=sample_name)
print(organize_x(params,len(params)))
