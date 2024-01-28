"""
RBB: Assimilative Social Influence, Nomina Opinions (e.g., my connection adapted/did not adapt)
Develop an RBB that represents assimilative social influence with nominal opinions affecting flood adaptation behaviors
in a neighborhood.
https://jasss.soc.surrey.ac.uk/20/4/2.html -> refer to sections 2.15-2.18
"""

# TODO: change flood logic st stop after flood t.s. 15
# TODO: after flood, check lost cost due to decisions/adaptations
# TODO: track n of adaptations -> change datacollector logic
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
import networkx as nx
from tqdm import tqdm
import pandas as pd
from scipy.stats import truncnorm, norm
import numpy as np
import concurrent.futures



def run_model(p, n=100, plot=False, steps=15):
    model = AdaptationModel(seed=0, number_of_households=n, flood_map_choice="harvey", polarization=p, threshold=0.5)
    for step in range(steps):
        if plot: model.plot_network(big=True, labels=True)
        model.step()
    if plot: model.plot_network(big=True, labels=True)
    return model, model.datacollector.get_agent_vars_dataframe(), model.datacollector.get_model_vars_dataframe()

def run_simulation(p):
    arr = []
    for i in range(500):
        model, agent_df, model_df = run_model(p, n=100, plot=False)
        arr.append(model_df.iloc[-1]["winners"])
    return p, arr

def plot_data(p, arr):
    N, BINS, _ = plt.hist(arr)
    mu, std = norm.fit(arr)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(x, mu, std)
    scaling_factor = N.sum() * np.diff(BINS)[0]
    plt.plot(x, scaling_factor * pdf, 'k', linewidth=2)
    plt.title(f"p={p}")
    plt.savefig(f"hist_{p}.png")
    plt.close()
    return p, mu, std

def run_batch():
    res = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, p) for p in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 95]]

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                p, arr = future.result()
                res.append(plot_data(p, arr))
                pbar.update(1)

    np.savetxt("results.csv", np.array(res), delimiter=",")

if __name__ == "__main__":
    run_batch()
