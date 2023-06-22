import pandas as pd
from calculate_experimental_results import D0calc_MonteCarloErrors
import os

def generate_inputs(nameOfInputCSVFile, nameOfExperimentalResultsFile):


    expData = pd.read_csv(nameOfInputCSVFile,header=None)
        
    #If extra columns get read in, trim them down to just 3
    if expData.shape[1] >=4:
        expData = expData.loc[:,1:4]
        
    # Name the columsn of the iput data
    expData.columns = ["TC", "thr","M", "delM"]

    # Calculate Daa from experimental results
    expResults = D0calc_MonteCarloErrors(expData)

    # Combine the diffusion parameters with the experimental setup (T, thr, M, delM)
    # to get a final dataframe that will be passed into the optimizer
    diffusionExperimentResults = expData.join(expResults)

    # Write dataframe to a .csv file
    diffusionExperimentResults.to_csv(nameOfExperimentalResultsFile)


#main

#generate some results
dir_path = os.path.dirname(os.path.realpath(__file__))

nameOfInputCSVFile =  f"{dir_path}/data/KM95-28-Dc-1250um.csv"
nameOfExperimentalResultsFile = f"{dir_path}/data/input_KM95-28-Dc-1250um.csv"
generate_inputs(nameOfInputCSVFile, nameOfExperimentalResultsFile)