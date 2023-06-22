
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
data_input = pd.read_csv(f"{dir_path}/data/input_KM95-28-Dc-1250um.csv")
domains_to_model = 3
mineral_name = "quartz"
time_add = [300*60,110073600]
temp_add = [40,21.111111111111]
sample_name = "KM95-28-Dc-1250um"
moves = "snooker" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists
misfit_stat = "l2_moles"
omit_value_indices = []#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]




#################################################################

def organize_x(x,ndim):
        ndom = int(((ndim-1)/2))
        moles = x[0]
        Ea = x[1]
        lnd0aa = x[2:2+ndom]
        fracs = x[2+ndom:]
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


params, misfit_val = diffEV_multiples(objective,dataset,5,mineral_name,domains_to_model)
start_time = time.time()


plot_results(params,dataset,objective,sample_name=sample_name)
