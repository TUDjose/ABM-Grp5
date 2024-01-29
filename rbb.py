# import numpy as np
# from scipy.stats import truncnorm
#
# def nominal_opinions(model: mesa.Model, network: nx.DiGraph) -> None:
#     """
#     Method used to set the opinions of the agents in the model. Nominal opinions are set based on the equation:
#         O[i, t+1] = Max(Sum(w[i,j]) for each O[j,t])
#     where: O[i,t+1] is the opinion of agent i at time t+1
#            w[i,j] is the weight of the edge between agent i and agent j
#            O[j,t] is the opinion of neighboring agent j at time t
#     O[i, t] is the opinion of agent i at time t, considered to be either -1, 0 or 1
#
#     params:
#         mesa_model: the mesa model with the agents containing a nominal/discrete opinion parameter
#         network: directional graph connecting the agents in the model with weighted edges
#     """
#
#     new_opinions: list[int] = []
#     for agent in model.schedule.agents:  # iterate over each agent in the model to calculate new opinion
#         nbor_sums: dict[int, float] = {}  # dictionary to store the sum of weights for each opinion
#         neighbors: list[Households] = [model.schedule.agents[edge[1]] for edge in network.out_edges(agent.unique_id)]
#         for nbor in neighbors:
#             if nbor_sums.get(nbor.opinion) is None:
#                 nbor_sums[nbor.opinion] = network[nbor.unique_id][agent.unique_id]['weight']
#             else:
#                 nbor_sums[nbor.opinion] += network[nbor.unique_id][agent.unique_id]['weight']
#
#         most_common = max(nbor_sums, key=lambda k: nbor_sums[k])  # calculate the most common opinion among neighbors
#         new_opinions.append(most_common)
#
#     for i, agent in enumerate(model.schedule.agents):
#         agent.opinion = new_opinions[i]  # update the opinion of each agent after having calculated all new opinions