import math
import numpy as np
import pandas as pd
import torch
import math as math
from jax import numpy as jnp

def forwardModelKineticsDiffEV(kinetics, lookup_table,tsec,TC, geometry:str = "spherical"): 


    # Check the number of dimensions being passed in to see how many vectors we're dealing with. Code handles 1 vs >1 differently
    if kinetics.ndim > 1:
        num_vectors = len(kinetics[0,:])
    else:
        num_vectors = 1
    

    # Infer the number of domains from input
    if len(kinetics) <= 3:
        ndom = 1
    else:
        ndom = (len(kinetics))//2

    # Convert to a tensor for speed

    kinetics = torch.tensor(kinetics)
    Ea = kinetics[0] # Moles isn't passed into this function, so first entry of kinetics is Ea
    kinetics = kinetics[1:] 
    temp = kinetics[1:]
    # kinetics: (Ea, lnd0aa_x, fracs_x). To make this compatible with other functions, if there are x fracs, input x-1 fractions, and the code will determine the
    # final fraction.

    R = 0.008314 #gas constant
    torch.pi = torch.acos(torch.zeros(1)).item() * 2


    if num_vectors == 1:
  
        lnD0aa = torch.tile(kinetics[0:ndom].T,(len(TC),1)) # Do this for LnD0aa
        fracstemp = kinetics[ndom:] # Grab fracs that were input (one will be missing because it is pre-determined by the others)

        fracs = torch.tile(torch.concat((fracstemp,1-torch.sum(fracstemp,axis=0,keepdim=True)),axis=0).T,(len(TC),1)) # Add the last frac as 1-sum(other fracs)
        Ea = torch.tile(Ea,(len(TC),ndom)) # Do for Ea

    


        # Put time and cumulative time in the correct shape
        if ndom > 1:
            tsec = torch.tile(torch.reshape(tsec,(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of getting tsec into a numdom x numstep matrix for multiplication
            cumtsec = torch.tile(torch.reshape(torch.cumsum(tsec[:,1],dim=0),(-1,1)),(1,Ea.shape[1])) #Same as above, but for cumtsec        
            # Convert TC to TK and put in correct shape for quick computation                                                 
            TK = torch.tile(torch.reshape((TC + 273.15),(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of turning TC from a 1-d array to a 2d array and making two column copies of it

        else:
            cumtsec = torch.reshape(torch.cumsum(tsec,-1),(-1,1))
            TK = torch.reshape(TC+273.15,(-1,1))
            tsec = torch.reshape(tsec,(-1,1))

        # Calculate D/a^2 for each domain

        Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))

        # Pre-allocate fraction and Dtaa
        f = torch.zeros(Daa.shape)
        Dtaa = torch.zeros(Daa.shape)
        DtaaForSum = torch.zeros(Daa.shape)


        # Calculate Dtaa in incremental (not cumulative) form including the added heating steps

        DtaaForSum[0,:] = Daa[0,:]*tsec[0,:]
        DtaaForSum[1:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])

        if geometry == "spherical":
            # # Make the correction for P_D vs D_only
            for i in range(len(DtaaForSum[0,:])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
                if DtaaForSum[0,i] <= 1.347419e-17:
                    DtaaForSum[0,i] *= 0
                elif DtaaForSum[0,i] >= 4.698221e-06:
                    pass
                else:
                    DtaaForSum[0,i] *= lookup_table(DtaaForSum[0,i])

            # Calculate Dtaa in cumulative form.
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)


            # Calculate f at each step
            Bt = Dtaa*torch.pi**2
            f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
            f[Bt>0.0091] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[Bt>0.0091])-(3/(torch.pi**2))* \
                    ((torch.pi**2)*Dtaa[Bt>0.0091])
            f[Bt >1.8] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.8])
            


            # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
        elif geometry == "plane sheet":
            # Need to derive a correction for the plane sheet... for now I just won't do an irradiation correction
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)
            f = (2/np.sqrt(math.pi))*np.sqrt((Dtaa))
            f[f > 0.6] = 1-(8/(math.pi**2))*np.exp(-math.pi**2*Dtaa[f > 0.6]/4)

        f_MDD = f*fracs

        # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
        # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
        sumf_MDD = torch.sum(f_MDD,axis=1)

        # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
        # Return that sumf_MDD == 0
        if (torch.round(sumf_MDD[2],decimals=6) == 1):


            return torch.zeros(len(sumf_MDD)-2)
            

        # Remove the two steps we added, recalculate the total sum, and renormalize.
        newf = torch.zeros(sumf_MDD.shape)
        newf[0] = sumf_MDD[0]
        newf[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]

        newf = newf[2:]
        normalization_factor = torch.max(torch.cumsum(newf,0))

        punishmentFlag = torch.round(sumf_MDD[-1],decimals=3) < 1.0
        #punishmentFlag = torch.round(newf[-1,:],decimals = 5) < 1
        
        diffFi= newf/normalization_factor 


        # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
        # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
        # special case when i = 1; need to insert 0 for previous amount released



        # Resum the gas fractions into cumulative space that doesn't include the two added steps
        sumf_MDD = torch.cumsum(diffFi,axis=0)
  
        return sumf_MDD, punishmentFlag 


    else:
        lnD0aa = kinetics[0:ndom].unsqueeze(0).expand(len(TC), ndom, -1)
        fracstemp = kinetics[ndom:]
        fracs = torch.cat((fracstemp, 1 - torch.sum(fracstemp, axis=0, keepdim=True))).unsqueeze(0).expand(len(TC), -1, -1)
        Ea = Ea.unsqueeze(0).expand(len(TC),ndom,-1)



    # THIS IS TEMPORARY-- WE NEED TO ADD THIS AS AN INPUT.. THE INPUTS WILL NEED TO BE
    # 1. Duration of irradiation
    # 2. Temperature during irradiation
    # 3. Duration of lab storage
    # 4. Temperature during lab storage

    # We might also want to make this all optional at some point, since some minerals are so retentive 
    # that they wont lease any helium during irradiation and storage.


        if ndom > 1:
            cumtsec = torch.cumsum(tsec,dim=0).unsqueeze(-1).repeat(1,ndom) # Expand dimensions to match the desired shape
            cumtsec = cumtsec.unsqueeze(2).repeat(1,1,num_vectors)

            TK = (TC+273.15).unsqueeze(-1).repeat(1,ndom)
            TK = TK.unsqueeze(2).repeat(1,1,num_vectors)

            tsec = tsec.unsqueeze(-1).repeat(1,ndom)
            tsec = tsec.unsqueeze(2).repeat(1,1,num_vectors)

            # Expand lnD0aa to match the desired shape



        else:

            cumtsec = torch.cumsum(tsec,dim=0).unsqueeze(-1).repeat(1,ndom) # Expand dimensions to match the desired shape
            cumtsec = cumtsec.unsqueeze(2).repeat(1,1,num_vectors)

            TK = (TC+273.15).unsqueeze(-1).repeat(1,ndom)
            TK = TK.unsqueeze(2).repeat(1,1,num_vectors)

            tsec = tsec.unsqueeze(-1).repeat(1,ndom)
            tsec = tsec.unsqueeze(2).repeat(1,1,num_vectors)

            # Expand lnD0aa to match the desired shape


        # Calculate D/a^2 for each domain
        Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))
        # Pre-allocate fraction and Dtaa
        f = torch.zeros(Daa.shape)
        Dtaa = torch.zeros(Daa.shape)
        DtaaForSum = torch.zeros(Daa.shape)


        # Calculate Dtaa in incremental (not cumulative) form including the added heating steps
        if num_vectors > 1:

            DtaaForSum[0,:,:] = Daa[0,:,:]*tsec[0,:,:]
            DtaaForSum[1:,:,:] = Daa[1:,:,:]*(cumtsec[1:,:,:]-cumtsec[0:-1,:,:])
        else:
            DtaaForSum[0,:] = Daa[0,:]*tsec[0,:]
            DtaaForSum[1:,:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])
        if geometry == "spherical":
            # Make the correction for P_D vs D_only
            for j in range(len(DtaaForSum[0,0,:])    ):
                for i in range(len(DtaaForSum[0,:,0])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
                    if DtaaForSum[0,i,j] <= 1.347419e-17:
                        DtaaForSum[0,i,j] *= 0
                    elif DtaaForSum[0,i,j] >= 4.698221e-06:
                        pass
                    else:
                        DtaaForSum[0,i,j] *= lookup_table(DtaaForSum[0,i,j])

            # Calculate Dtaa in cumulative form.
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)


            # Calculate f at each step
            Bt = Dtaa*torch.pi**2

            f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
            f[Bt>0.0091] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[Bt>0.0091])-(3/(torch.pi**2))* ((torch.pi**2)*Dtaa[Bt>0.0091])
            f[Bt >1.8] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.8])

        elif geometry == "plane sheet":
            # Need to derive a correction for the plane sheet... for now I just won't do an irradiation correction
            Dtaa = torch.cumsum(DtaaForSum, axis = 0)
            
            f = (2/np.sqrt(math.pi))*np.sqrt((Dtaa))
            f[f > 0.6] = 1-(8/(math.pi**2))*np.exp(-math.pi**2*Dtaa[f > 0.6]/4)
            

        # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
        f_MDD = f*fracs

        # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
        # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
        sumf_MDD = torch.sum(f_MDD,axis=1)

        # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
        # Return that sumf_MDD == 0
        if num_vectors == 1:
        
            if (torch.round(sumf_MDD[2],decimals=6) == 1):
                return torch.zeros(len(sumf_MDD)-2)
            


        newf = torch.zeros(sumf_MDD.shape)
        newf[0] = sumf_MDD[0]
        newf[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]

        newf = newf[2:]

        normalization_factor = torch.max(torch.cumsum(newf,0),axis=0).values
    
        punishmentFlag = torch.round(sumf_MDD[-1,:],decimals = 3) < 1

        diffFi= newf/normalization_factor 




        # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
        # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
        # special case when i = 1; need to insert 0 for previous amount released

        # has_positive_value = (tensor > 0).any()

        # # Add a breakpoint if positive values are present
        # if has_positive_value:
        #     breakpoint()

        # Resum the gas fractions into cumulative space that doesn't include the two added steps
        sumf_MDD = torch.cumsum(diffFi,axis=0)
        nan_mask = torch.isnan(sumf_MDD).all(dim=0)
        sumf_MDD[:,nan_mask]= 0.0

     
        return sumf_MDD,punishmentFlag
    



def forwardModelKineticsIpopt(kinetics,lookup_table,tsec,TC): 
# This is a version of the function that is compatible with the ipopt diffusion optimizer

# kinetics: (Ea, lnd0aa_x, fracs_x). To make this compatible with other functions, if there are x fracs, input x-1 fractions, and the code will determine the
# final fraction.

    R = 0.008314 #gas constant
    #pi = math.acos(jnp.zeros(1)) * 2

    # Infer the number of domains from input
    if len(kinetics) <= 3:
        ndom = 1
    else:
        ndom = (len(kinetics))//2

    # Make a subset of X, removing the Ea so that we have an even number of elements
    temp = kinetics[1:]

    breakpoint()
    lnD0aa = jnp.tile(temp[0:ndom],(len(TC),1)) # Do this for LnD0aa
    breakpoint()
    fracstemp = temp[ndom:] # Grab fracs that were input (one will be missing because it is pre-determined by the others)
    fracs = jnp.tile(jnp.concatenate((fracstemp,1-jnp.sum(fracstemp,axis=0,keepdims=True)),axis=-1),(len(TC),1)) # Add the last frac as 1-sum(other fracs)
    Ea = jnp.tile(kinetics[0],(len(TC),ndom)) # Do for Ea



    # Put time and cumulative time in the correct shape
    if ndom > 1:
        tsec = jnp.tile(jnp.reshape(tsec,(-1,1)),(1,Ea.shape[1])) # This is a complicated-looking way of getting tsec into a numdom x numstep matrix for multiplication
        cumtsec = jnp.tile(jnp.reshape(jnp.cumsum(tsec[:,1],axis=0),(-1,1)),(1,Ea.shape[1])) # Same as above, but for cumtsec 

        # Convert TC to TK and put in correct shape for quick computation                                                 
        TK = jnp.tile(jnp.reshape((TC + 273.15),(-1,1)),(1,Ea.shape[1]))


    else:
        cumtsec = jnp.reshape(jnp.cumsum(tsec,-1),(-1,1))
        TK = jnp.reshape(TC+273.15,(-1,1))
        tsec = jnp.reshape(tsec,(-1,1))

    # Calculate D/a^2 for each domain
    Daa = jnp.exp(lnD0aa)*jnp.exp(-Ea/(R*TK))

    # Pre-allocate fraction and Dtaa
    f = jnp.zeros(Daa.shape)
    Dtaa = jnp.zeros(Daa.shape)
    DtaaForSum = jnp.zeros(Daa.shape)
    

    # Calculate Dtaa in incremental (not cumulative) form including the added heating steps

    DtaaForSum = (Daa[0,:]*tsec[0,:]).reshape(1,ndom)
    DtaaForSum = jnp.vstack([DtaaForSum,Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])])
    #DtaaForSum[1:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])

    # I NEED TO ADJUST THIS FOR JAX STILL! IT'S FINE TO HAVE IT OFF FOR TESTING SINCE IT'S A SMALL CORRECTION.
    # for i in range(len(DtaaForSum[0,:])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
    #     if DtaaForSum[0,i] <= 1.347419e-17:
    #         DtaaForSum[0,i] *= 0
    #     elif DtaaForSum[0,i] >= 4.698221e-06:
    #         pass
    #     else:
    #         DtaaForSum[0,i] *= lookup_table(DtaaForSum[0,i])

    # new_array = jnp.array([])
    # print(DtaaForSum[0])
    # for value in DtaaForSum[0]:
    #     if value <= 1.347419e-17:

    #         new_array = jnp.append(new_array,0)
    #     elif value < 4.698221e-06:

    #         breakpoint()
    #         new_array = jnp.append(new_array, value*jnp.array(lookup_table(value)))
    #         print("it worked")
    #     else:
    #         new_array = jnp.append(new_array, value)

    # To make this work, I just need to 

    # print(new_array)
    # Calculate Dtaa in cumulative form.
    Dtaa = jnp.cumsum(DtaaForSum, axis = 0)

    
    # Calculate f at each step
    Bt = Dtaa*math.pi**2

    Dtaa = jnp.cumsum(DtaaForSum, axis = 0)

    second_split = Bt>0.0091
    third_split = Bt > 1.8



    #kps_camera = kps_world - jnp.where(selected_rows[:,None], self.pos, 0) #jnp.where 
    

    f = (6/(jnp.pi**(3/2)))*jnp.sqrt((jnp.pi**2)*Dtaa)

    f = jnp.where(second_split,
            (6/(jnp.pi**(3/2)))*jnp.sqrt((jnp.pi**2)*Dtaa)-(3/(jnp.pi**2))*((jnp.pi**2)*Dtaa), 
            f)
    f = jnp.where(third_split,
            1 - (6/(jnp.pi**2))*jnp.exp(-(jnp.pi**2)*Dtaa),
            f)


    

    # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
    f_MDD = f*fracs

    # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
    # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
    sumf_MDD = jnp.sum(f_MDD,axis=1)

    # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
    # Return that sumf_MDD == 0
    if (jnp.round(sumf_MDD[2],decimals=6) == 1):
        return jnp.zeros(len(sumf_MDD)-2)
        

    # Remove the two steps we added, recalculate the total sum, and renormalize.


    newf = sumf_MDD[0]
    newf = jnp.append(newf, sumf_MDD[1:]-sumf_MDD[0:-1])
    newf = newf[2:]
    normalization_factor = jnp.max(jnp.cumsum(newf,0))
    diffFi= newf/normalization_factor 

    # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
    # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
    # special case when i = 1; need to insert 0 for previous amount released


    
    # Resum the gas fractions into cumulative space that doesn't include the two added steps
    sumf_MDD = jnp.cumsum(diffFi,axis=0)

    return sumf_MDD



