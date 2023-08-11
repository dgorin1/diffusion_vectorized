
from forwardModelKinetics import forwardModelKineticsDiffEV
from forwardModelKinetics import forwardModelKineticsIpopt
from dataset import Dataset
import math as math 
import pickle
import jax
import torch as torch
import numpy as np
from jax import numpy as jnp


class DiffusionObjective():
    def __init__(self, method:str,data:Dataset, time_add: jnp.array, temp_add: jnp.array,pickle_path="../lookup_table.pkl",omitValueIndices = [], stat: str = "chisq", geometry:str = "spherical"):
        self.method = method
        if self.method == "ipopt":
            self.dataset = data
            self.lookup_table = pickle.load(open(pickle_path,'rb'))
            self.time_add = time_add
            self.temp_add = temp_add

            # Add total moles information for priors
            self.total_moles = torch.sum(torch.tensor(self.dataset.M))
            self.total_moles_del = torch.sqrt(torch.sum(torch.tensor(self.dataset.delM)**2))

            #self.omitValueIndices = jnp.array(omitValueIndices)
                  
            time = self.dataset._thr*3600
            self.tsec = jnp.concatenate([time_add,time])
            self._TC = jnp.concatenate([temp_add,self.dataset._TC])
            #(~np.isin(jnp.arange(len(self.dataset)), jnp.array(omitValueIndices))).astype(int)
            self.omitValueIndices = (~np.isin(jnp.arange(len(self.dataset)), jnp.array(omitValueIndices))).astype(int)
            #(~jnp.isin(jnp.arange(len(self.dataset)), jnp.array(omitValueIndices)).astype(int)
            self.grad = jax.grad(self.objective)
        
        elif self.method == "MCMC" or self.method == "diffEV":
            self.dataset = data
            self.lookup_table = pickle.load(open(pickle_path,'rb'))
            self.time_add = time_add
            self.temp_add = temp_add

            # Add total moles information for priors
            self.total_moles = torch.sum(torch.tensor(self.dataset.M))
            self.total_moles_del = torch.sqrt(torch.sum(torch.tensor(self.dataset.delM)**2))

            #self.omitValueIndices = jnp.array(omitValueIndices)
            self.stat = stat
            time = self.dataset._thr*3600
            self.tsec = torch.cat([time_add,time])
            self._TC = torch.cat([temp_add,self.dataset._TC])
            self.omitValueIndices = torch.isin(torch.arange(len(self.dataset)), torch.tensor(omitValueIndices)).to(torch.int)
            self.plateau = torch.sum(((torch.tensor(self.dataset.M)-torch.zeros(len(self.dataset.M)))**2)/(data.uncert**2))
            self.Fi = torch.tensor(data.Fi)
            self.geometry = geometry

    
    
    def __call__(self, X):

        return self.objective(X)
    
    
    def grad(self, X):
        return self.grad(X)

    
    def objective(self, X): #__call__ #evaluate

        data = self.dataset
        # This function calculates the fraction of gas released from each domain
        # in an MDD model during the heating schedule used in the diffusion
        # experiment. Then the fractions released from each domain are combined in
        # proportion to one another as specified by the MDD model, and the
        # diffusivity of each step is calculated. A residual is calculated as the
        # sum of absolute differences between the observed and modeled release
        # fractions over all steps.

        # JOSH, DO I GET TO ASSUME THAT X IS GOING TO COME IN AS A JAX NP ARRAY?
        # X = jnp.array(X)
 
        if len(X) % 2 != 0:
            total_moles = X[0]
            X = X[1:]
        
        if len(X) <= 3:
            ndom = 1
        else:
            ndom = (len(X))//2

        # Grab the other parameters from the input
        temp = X[1:]

        if self.method == "ipopt":
            Fi_MDD = forwardModelKineticsIpopt(X,self.lookup_table,self.tsec,self._TC) # Gas fraction released for each heating step in model experiment
            Fi_exp = data.Fi_exp #Gas fraction released for each heating step in experiment
        
            # Calculate the Fraction released for each heating step in the real experiment
            TrueFracFi = Fi_exp[1:] - Fi_exp[:-1]
            TrueFracFi = jnp.concatenate((jnp.expand_dims(Fi_exp[0], axis=-1), TrueFracFi), axis=-1)


            # Calculate the fraction released for each heating step in the modeled experiment
            trueFracMDD = Fi_MDD[1:] - Fi_MDD[:-1]
            trueFracMDD = jnp.concatenate((jnp.expand_dims(Fi_MDD[0], axis=-1), trueFracMDD), axis=-1)
            
                    # Sometimes the forward model predicts kinetics such that ALL the gas would have leaked out during the irradiation and lab storage.
            # In this case, we end up with trueFracMDD == 0, so we should return a high misfit because we know this is not true, else we wouldn't
            # have measured any He in the lab. 

            exp_moles = data._M
            moles_MDD = trueFracMDD * total_moles
            if jnp.sum(trueFracMDD) == 0:
                moles_MDD = jnp.zeros([moles_MDD.shape[0]])

            if jnp.isnan(moles_MDD).all():
                moles_MDD = jnp.zeros([moles_MDD.shape[0]])

            misfit = ((exp_moles-moles_MDD)**2)/(data.uncert**2)

            return jnp.sum(misfit, where = self.omitValueIndices)



        elif self.method == "diffEV":
            # Forward model the results so that we can calculate the misfit.

            Fi_MDD,punishmentFlag = forwardModelKineticsDiffEV(X,self.lookup_table,self.tsec,self._TC,geometry = self.geometry)
            punishmentFlag = punishmentFlag *100 + 1

            exp_moles = torch.tensor(data.M)
            if len(X.shape) > 1:

                if X.shape[1] == 0: #If we get passed an empty vector, which seems to happen when all generated samples do not meet constraints
                    return([])

                        # Calculate the fraction released for each heating step in the modeled experiment
                elif X.shape[1] ==1: 
                    trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
                    trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-0),trueFracMDD),dim=-1)

                    if self.stat == "l1_frac" or self.stat == "l2_frac" or self.stat == "percent_frac":
                        trueFracFi = self.Fi[1:] - self.Fi[0:-1]
                        trueFracFi = torch.concat((torch.unsqueeze(self.Fi[0],dim=-0),trueFracFi),dim=-1)
                    else:
                        moles_MDD = trueFracMDD * total_moles

                    # Scale by chosen number of moles



                    if self.stat.lower() == "chisq":
                        misfit = torch.sum((1-self.omitValueIndices)*((exp_moles-moles_MDD)**2)/(data.uncert**2))
                    elif self.stat.lower() == "l1_moles":
                        misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(exp_moles-moles_MDD)))
                    elif self.stat.lower() == "l2_moles":
                        misfit = torch.sum((1-self.omitValueIndices)*((exp_moles-moles_MDD)**2))
                    elif self.stat.lower() == "l1_frac":
                        misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(trueFracFi-trueFracMDD)))
                    elif self.stat.lower() == "l2_frac":
                        misfit = torch.sum((1-self.omitValueIndices)*(trueFracFi-trueFracMDD)**2)
                    elif self.stat.lower() == "percent_frac":

                        misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(trueFracFi-trueFracMDD))/trueFracFi)
                    



                else:
                    trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
                    trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=0),trueFracMDD),dim=0)

                    if self.stat.lower() == "l1_frac" or self.stat.lower() == "l2_frac" or self.stat.lower() == "percent_frac":
                        trueFracFi = self.Fi[1:] - self.Fi[0:-1]
                        trueFracFi = torch.concat((torch.unsqueeze(self.Fi[0],dim=-0),trueFracFi),dim=-1)
                        trueFracFi = torch.tile(trueFracFi.unsqueeze(1),[1,trueFracMDD.shape[1]])

                    else:
                        moles_MDD = trueFracMDD * total_moles
                    
                    multiplier = 1- torch.tile(self.omitValueIndices.unsqueeze(1),[1,trueFracMDD.shape[1]])
                    
                    
                    if self.stat.lower() == "chisq":
                        misfit = torch.sum(multiplier*((exp_moles.unsqueeze(1)-moles_MDD)**2)/(data.uncert.unsqueeze(1)**2),axis=0)
                    elif self.stat.lower() == "l1_moles":
                        misfit = misfit = torch.sum(multiplier*(torch.abs(exp_moles.unsqueeze(1)-moles_MDD)),axis=0)
                    elif self.stat.lower() == "l2_moles":
                        misfit = torch.sum((multiplier*((exp_moles.unsqueeze(1)-moles_MDD)**2)),axis=0)
                    elif self.stat.lower() == "l1_frac":
                        misfit = torch.sum(multiplier*(torch.abs(trueFracFi-trueFracMDD)),axis=0)
                    elif self.stat.lower() == "l2_frac":
                        misfit = torch.sum((multiplier*(trueFracFi-trueFracMDD)**2),axis=0)
                    elif self.stat.lower() == "percent_frac":

                        misfit = torch.sum(multiplier*(torch.abs(trueFracFi-trueFracMDD))/trueFracFi,axis=0)

                return misfit*punishmentFlag
                

            trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
            trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-0),trueFracMDD),dim=-1)
    
            if self.stat == "l1_frac" or self.stat == "l2_frac" or self.stat == "percent_frac":
                trueFracFi = self.Fi[1:] - self.Fi[0:-1]
                trueFracFi = torch.concat((torch.unsqueeze(self.Fi[0],dim=-0),trueFracFi),dim=-1)
            else:
                moles_MDD = trueFracMDD * total_moles



            if self.stat.lower() == "chisq":
                misfit = torch.sum(((1-self.omitValueIndices)*(exp_moles-moles_MDD)**2)/(data.uncert**2))
            elif self.stat.lower() == "l1_moles":
                misfit = torch.sum((1-self.omitValueIndices)*torch.abs(exp_moles-moles_MDD))
            elif self.stat.lower() == "l2_moles":
                misfit = torch.sum((1-self.omitValueIndices)*((exp_moles-moles_MDD)**2))
            elif self.stat.lower() == "l1_frac":
                misfit = torch.sum((1-self.omitValueIndices)*torch.abs(trueFracFi-trueFracMDD))
            elif self.stat.lower() == "l2_frac":
                misfit = torch.sum((1-self.omitValueIndices)*(trueFracFi-trueFracMDD)**2)
            elif self.stat.lower() == "percent_frac":
                misfit = torch.sum((1-self.omitValueIndices)*(torch.abs(trueFracFi-trueFracMDD))/trueFracFi)

            return misfit*punishmentFlag
