"""
RBB: Assimilative Social Influence, Nomina Opinions (e.g., my connection adapted/did not adapt)
Develop an RBB that represents assimilative social influence with nominal opinions affecting flood adaptation behaviors
in a neighborhood.
https://jasss.soc.surrey.ac.uk/20/4/2.html -> refer to sections 2.15-2.18
"""
import numpy as np

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


def plot_network(ax, model):
    pos = nx.spring_layout(model.G)
    ax.clear()
    colors = ['blue' if agent.opinion == 1 else 'red' for agent in model.schedule.agents]
    nx.draw(model.G, pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_title(f"Social Network State at Step {model.schedule.steps}", fontsize=12)
    # labels = nx.get_edge_attributes(model.G, 'weight')
    # nx.draw_networkx_edge_labels(model.G, pos, edge_labels=labels)
    plt.show()

def run_model(p, n=100, plot=False, p_seed=0):
    model = AdaptationModel(seed=0, number_of_households=n, flood_map_choice="harvey", network="watts_strogatz", polarization=p, p_seed=p_seed)
    for step in range(7):
        model.step()
        if plot:
            fig, ax = plt.subplots(figsize=(7, 7))
            plot_network(ax, model)

    return model, model.datacollector.get_agent_vars_dataframe(), model.datacollector.get_model_vars_dataframe()


if __name__ == "__main__":
    M, a, m = run_model(0.5, n=100, plot=True)


    # with tqdm(total=500*11) as pbar:
    #     for p in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, .8, .9, 95]:
    #         arr = []
    #         for i in range(500):
    #             model, agent_df, model_df = run_model(0.05, n=100, plot=False, p_seed=i)
    #             arr.append(model_df.iloc[-1]["total_positive_opinions"])
    #             pbar.update(1)
    #         N, BINS, _ = plt.hist(arr)
    #         mu, std = norm.fit(arr)
    #         print(p, mu, std)
    #         xmin, xmax = plt.xlim()
    #         x = np.linspace(xmin, xmax, 100)
    #         pdf = norm.pdf(x, mu, std)
    #         scaling_factor = N.sum() * np.diff(BINS)[0]
    #         plt.plot(x, scaling_factor*pdf, 'k', linewidth=2)
    #         plt.title(f"p={p}")
    #         plt.savefig(f"hist_{p}.png")
    #         plt.show()
