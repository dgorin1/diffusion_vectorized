
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
data_input = pd.read_csv(f"{dir_path}/data/input_n13ksp_moles_plane_sheet.csv")
mineral_name = "kspar"
time_add = [] #Add extra time in seconds
temp_add = [] # Add temps in C
sample_name = "Harrison values"
moves = "snooker" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists
max_domains_to_model = 8
geometry  = "plane sheet" #"plane sheet" # options are "plane sheet", or "spherical"
omit_value_indices =  [33,34,35,36,37,38,39,40,41,42,43]

misfit_stat_list = ["percent_frac","l1_frac","l2_frac","l2_moles","lnd0aa","percent_frac","chisq","l1_moles","l2_moles","l1_frac","l2_frac"] #options are chisq, l1_moles, l2_moles, l1_frac, l2_frac, percent_frac



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
    domains_to_model = 8
    
    save_params = np.empty((max_domains_to_model-1,max_domains_to_model*2+4))
    save_params.fill(np.NaN)

        
        
    print(f"{misfit_stat} with {domains_to_model} domains")

    dataset = Dataset("diffEV", data_input)


    objective = DiffusionObjective(
        "diffEV",
        dataset, 
        time_add = torch.tensor(time_add), 
        temp_add = torch.tensor(temp_add), 
        pickle_path = f"{dir_path}/data/lookup_table.pkl",
        omitValueIndices= omit_value_indices,
        stat = misfit_stat,
        geometry = "plane sheet"
    )

    # Read in the nonlinear constraint


    
    params = np.array([195.2191933,	19.44830467,16.17659628,13.93007269,8.893664002,8.676650883,6.212899801,6.17580078,	2.470272004,0.02078564,	0.07973485,	0.06776753,	0.18618525,	0.10420018,	0.22466006,	0.13960004])
    #params = np.array([0.000000000005210,	267.373350300723000,	32.554251749378100,	28.370979816929700	,25.461449986687800,	22.804036255907300,	20.173786810313400	,15.583851430783000,	13.182894713993100	,10.870667542461900	,0.012120858997346,	0.027280174405612	,0.061987256302431	,0.066114795944613	,0.043563780825043	,0.247719079568467,	0.168491995737490])
    
    temp = objective.objective(params)
    print(temp)
    breakpoint()
    plot_results(params,dataset,objective,sample_name=sample_name,quiet = True,misfit_stat = misfit_stat)
    #pickle_path = f"{dir_path}/data/lookup_table.pkl"
    #lookup_table = pickle.load(open(pickle_path,'rb'))
    #forwardModelKineticsDiffEV(params, lookup_table,tsec,TC)


        
        #print(organize_x(params,len(params),chop_fracs = False))
        #params = organize_x(params,len(params),chop_fracs = False)
        



