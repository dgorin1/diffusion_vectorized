import os
import datetime as datetime
import csv

def save_results(num_domains:int, sample_name:str = "",misfit_stat:str = "",params = [],moves_type = ""):

    # Get the current date and time
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{num_domains}domains_{misfit_stat}_{moves_type}_{current_date}"


    folder_name = os.path.join("MCMC_data", f"{sample_name}", run_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)



    # Generate the file name


    name = os.path.join(folder_name, f"{num_domains}domains_{current_date}_{misfit_stat}.csv")
    with open(name, 'w',newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(params)

    # Save the figure as a PDF file