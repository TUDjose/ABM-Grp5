"""
RBB: Assimilative Social Influence, Nominal Opinions (e.g., my connection adapted/did not adapt)
Develop an RBB that represents assimilative social influence with nominal opinions affecting flood adaptation behaviors
in a neighborhood.
https://jasss.soc.surrey.ac.uk/20/4/2.html -> refer to sections 2.15-2.18
"""
import networkx as nx

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
import os



def run_model(p, n, mn=5, plot=False, save=False, neutral=2.5):
    model = AdaptationModel(number_of_households=n, polarization=p, neutral_importance=neutral, max_neighbors=mn,
                            probability_of_network_connection=0.4)
    model.plot_network(big=False, labels=False) if plot else None
    for step in range(15):
        model.step()
    model.plot_network(big=False, labels=False) if plot else None
    # model.plot_model_domain_with_agents() if plot else None
    if save:
        dir = f'dataframes/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        model.datacollector.get_agent_vars_dataframe().to_csv(f'{dir}/agent_df.csv')
        model.datacollector.get_model_vars_dataframe().to_csv(f'{dir}/model_df.csv')

    return model, model.datacollector.get_agent_vars_dataframe(), model.datacollector.get_model_vars_dataframe(), nx.is_strongly_connected(model.G)

def run_simulation(p=1, nbors=5, house=100, neutral=2):
    NEUTRAL = []
    CONNECTED = []
    for _ in range(100):
        m = AdaptationModel(
            number_of_households=house,
            polarization=p,
            neutral_importance=neutral,
            max_neighbors=nbors
        )
        for _ in range(15):
            m.step()

        NEUTRAL.append(m.datacollector.get_model_vars_dataframe().iloc[-1]["neutral"])
        CONNECTED.append(1 if nx.is_strongly_connected(m.G) else 0)
    return p, nbors, house, neutral, NEUTRAL, CONNECTED

def run_batch():
    dir = f'results/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    res = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, p) for p in np.linspace(0.1, 1, 15)]

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                p, arr = future.result()
                res.append(np.concatenate((np.array([p]), [np.mean(arr)])))
                pbar.update(1)

    np.savetxt(f"{dir}/hist_data.csv", np.array(res), delimiter=",")
    analysis(f"{dir}/hist_data.csv")


def run_batch_2d():
    dir = f'results/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    for nbors in [2,3,4,5]:
        res = []
        with (concurrent.futures.ProcessPoolExecutor()
              as executor):
            futures = [executor.submit(run_simulation, p, nbors, 100, 2) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9, 1]]

            with tqdm(total=len(futures)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    P, NBORS, HOUSE, NEUTRAL, ARR1, ARR2 = future.result()
                    res.append([P, NBORS, HOUSE, NEUTRAL, np.mean(ARR1), np.sum(ARR2)/100])
                    pbar.update(1)

        np.savetxt(f"{dir}/data_pmn_{nbors}.csv", np.array(res), delimiter=",")

def analysis(dir):
    res = np.loadtxt(dir, delimiter=",")
    plt.scatter(res[:, 0], res[:, 1])
    z = np.polyfit(res[:, 0], res[:, 1], 1)
    p = np.poly1d(z)
    plt.plot(res[:, 0], p(res[:, 0]), "r--")
    plt.xlabel("Polarization")
    plt.ylabel("Neutral agents")
    plt.savefig('plots/results.png')
    plt.show()


def analysis_2d(dir, cols, min, max, x):
    df = pd.DataFrame(columns=cols, index=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    for filename in os.listdir(dir):
        data = np.loadtxt(dir + filename, delimiter=",")
        data = data[data[:, 0].argsort()]
        df[float(filename[min:max])] = data[:, 1]
    print(df)
    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(10, 10))
    plt.imshow(df, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Neutral agents")
    plt.xticks(np.arange(0, len(df.columns), 1), df.columns)
    plt.yticks(np.arange(0, len(df.index), 1), df.index)
    plt.xlabel(x)
    plt.ylabel("Polarization")
    plt.savefig(f'plots/{x}.png')


if __name__ == '__main__':
    # run_batch_2d()
    run_model(0.5, 100, neutral=1.8, mn=5, plot=True, save=True)



