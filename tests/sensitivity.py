'''
neutral importance
probability_of_network_connection
number_of_nearest_neighbours
'''
import pathlib
import concurrent.futures
import datetime
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from model import AdaptationModel

def run_simulation(polarization, neutral_importance, probability_of_network_connection, number_of_nearest_neighbours):
    ARR = []
    for _ in range(100):
        m = AdaptationModel(
            number_of_households=100,
            polarization=polarization,
            neutral_importance=neutral_importance,
            probability_of_network_connection=probability_of_network_connection,
            number_of_nearest_neighbours=number_of_nearest_neighbours,
        )
        for _ in range(15):
            m.step()
        ARR.append(m.datacollector.get_model_vars_dataframe().iloc[-1]["neutral"])
    return polarization, ARR

def neutral_importance_sensitivity():
    dir = f'results/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    for i in [1.1, 1.2, 1.3, 1.4]:
        res = []
        with (concurrent.futures.ProcessPoolExecutor()
              as executor):
            futures = [executor.submit(run_simulation, p, i, 0.4, 5) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]

            with tqdm(total=len(futures)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    p, arr = future.result()
                    res.append(np.concatenate((np.array([p]), [np.mean(arr)])))
                    pbar.update(1)

        np.savetxt(f"{dir}/data_importance_{i}.csv", np.array(res), delimiter=",")

def ponc_sensitivity():
    dir = f'results/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    for ponc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        res = []
        with (concurrent.futures.ProcessPoolExecutor()
              as executor):
            futures = [executor.submit(run_simulation, p, 2, ponc, 5) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]

            with tqdm(total=len(futures)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    p, arr = future.result()
                    res.append(np.concatenate((np.array([p]), [np.mean(arr)])))
                    pbar.update(1)

        np.savetxt(f"{dir}/data_ponc_{ponc}.csv", np.array(res), delimiter=",")

def nnn_sensitivity():
    dir = f'results/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    for nnn in [2,3,4,5,6,7,8,9,10]:
        res = []
        with (concurrent.futures.ProcessPoolExecutor()
              as executor):
            futures = [executor.submit(run_simulation, p, 2, 0.4, nnn) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]

            with tqdm(total=len(futures)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    p, arr = future.result()
                    res.append(np.concatenate((np.array([p]), [np.mean(arr)])))
                    pbar.update(1)

        np.savetxt(f"{dir}/data_nnn_{nnn}.csv", np.array(res), delimiter=",")





if __name__ == '__main__':
    neutral_importance_sensitivity()
    ponc_sensitivity()
    nnn_sensitivity()

