
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
from forwardModelKinetics import forwardModelKineticsDiffEV


# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_input = pd.read_csv(f"{dir_path}/data/input_n13ksp_moles.csv")
mineral_name = "kspar"
time_add = [0,0]
temp_add = [0,0]
sample_name = "n13ksp_lowerlimitOnlnd0aa"
moves = "snooker" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists
max_domains_to_model = 10
geometry  = "plane_sheet" # options are "plane_sheet", or "spherical"
omit_value_indices =  [35,36,37,38,39,40,41,42,43]

misfit_stat_list = ["percent_frac","chisq","l1_moles","l2_moles","l1_frac","l2_frac",] #options are chisq, l1_moles, l2_moles, l1_frac, l2_frac, percent_frac



def organize_x(x,ndim, chop_fracs = True):
        ndom = int(((ndim)/2))
        print(f"ndom is {ndom}")
        if len(x)%2 != 0:

            moles = x[0]
            x = x[1:]
        else:
             moles = np.NaN
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

        if "moles" in locals():

            output = np.append(moles,Ea)
        else:
             output = Ea
        output = np.append(output,lnd0aa)
        if chop_fracs == True:
            output = np.append(output,fracs[0:-1])
        else:
             output = np.append(output,fracs)
        return output

# Create dataset class for each associate package

for misfit_stat in misfit_stat_list:

    
    save_params = np.empty((max_domains_to_model-1,max_domains_to_model*2+4))
    save_params.fill(np.NaN)
    for i in range(2,max_domains_to_model+1):
        
        domains_to_model = i
        print(f"{misfit_stat} with {domains_to_model} domains")

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


        
        params = np.array([2.18E+02,	2.25E+01,	1.80E+01,	1.52E+01,	9.37E+00,	7.66E+00,	4.16E+0,6.13E-02,	8.10E-02,	1.03E-02,	5.19E-01,	2.11E-01])

        #pickle_path = f"{dir_path}/data/lookup_table.pkl"
        #lookup_table = pickle.load(open(pickle_path,'rb'))
        #forwardModelKineticsDiffEV(params, lookup_table,tsec,TC)


        objective.objective(params)
        #print(organize_x(params,len(params),chop_fracs = False))
        #params = organize_x(params,len(params),chop_fracs = False)
        



