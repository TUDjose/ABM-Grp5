"""
RBB: Assimilative Social Influence, Nominal Opinions (e.g., my connection adapted/did not adapt)
Develop an RBB that represents assimilative social influence with nominal opinions affecting flood adaptation behaviors
in a neighborhood.
https://jasss.soc.surrey.ac.uk/20/4/2.html -> refer to sections 2.15-2.18
"""
# TODO: sensitivity analysis

""""
MODEL VARS:
Cost of adaptation fixed value for all households: eg 10,000, from initial run without opinion propagation
polarization: 0-1, model dependent variable that represents the variance of the opinions weights (high P causes high variance)
weights between nodes: assigned based on polarization 

AGENT VARS:
opinion: -1 <-> 1, changes based on neighbors and weights
loss_tolerance: base + opinion * 1000  
    -> if damage > loss_tolerance + CoA, adapt (only occurs the time step right before flood)

EXPERIMENT:
    - fixed n of households (50, 100, ...)
    - total amount of flood damage to be the same each run
    - same cost of adaptation
    - same flood map
    - same seed flood damage
    - same network seed ???
    - vary polarization -> varies weights
    - vary radius of count_friends network
"""


from model import AdaptationModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import concurrent.futures
import datetime
import pathlib



def run_model(p, n, plot=False, save=False):
    model = AdaptationModel(number_of_households=n, polarization=p)
    model.plot_network(big=True, labels=False) if plot else None
    for step in range(15):
        model.step()
    model.plot_network(big=True, labels=False) if plot else None

    if save:
        dir = f'dataframes/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        model.datacollector.get_agent_vars_dataframe().to_csv(f'{dir}/agent_df.csv')
        model.datacollector.get_model_vars_dataframe().to_csv(f'{dir}/model_df.csv')

    return model, model.datacollector.get_agent_vars_dataframe(), model.datacollector.get_model_vars_dataframe()

def run_simulation(p):
    arr = []
    for i in range(250):
        model, agent_df, model_df = run_model(p=p, n=100, plot=False)
        arr.append(model_df.iloc[-1]["neutral"])
    return p, arr

def plot_data(p, arr, dir):
    N, BINS, _ = plt.hist(arr)
    plt.vlines(np.mean(arr), 0, N.max(), colors="r", linestyles="dashed")
    plt.title(f"p={p}")
    plt.savefig(f"{dir}/hist_{p}.png")
    plt.close()
    return p, np.mean(arr), np.std(arr)

def run_batch():
    dir = f'results/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    res = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                p, arr = future.result()
                res.append(plot_data(p, arr, dir))
                pbar.update(1)

    np.savetxt(f"{dir}/hist_data.csv", np.array(res), delimiter=",")

if __name__ == "__main__":
    model, A, M = run_model(p=.1, n=40, plot=True)
    print(M)
