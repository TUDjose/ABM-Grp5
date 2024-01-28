import numpy as np
from scipy.stats import truncnorm

def get_random_weight(mu):
    std = 0.15
    a, b = (0 - mu) / std, (1 - mu) / std
    w = truncnorm.rvs(a, b, loc=mu, scale=std)
    return np.around(w, 3)


def nominal_opinions(model, network, polarization, threshold):
    """
    Method used to set the opinions of the agents in the model. Nominal opinions are set based on the equation:
        O[i, t+1] = Max(Sum(w[i,j]) for each O[j,t])
    where: O[i,t+1] is the opinion of agent i at time t+1
           w[i,j] is the weight of the edge between agent i and agent j
           O[j,t] is the opinion of neighboring agent j at time t
    O[i, t] is the opinion of agent i at time t, considered to be either -1, 0 or 1

    params:
        mesa_model: the mesa model
        network: graph/network connecting the agents in the model with edges
        polarization: the polarization of the model
        threshold: the threshold below which the opinion of an agent can change according to the polarization
    """

    new_opinions = []
    for agent in model.schedule.agents:
        nbor_sums = {}
        neighbors = [model.schedule.agents[edge[1]] for edge in network.out_edges(agent.unique_id)]
        n = []
        for nbor in neighbors:
            n.append((nbor.unique_id, nbor.opinion))
            if nbor_sums.get(nbor.opinion) is None:
                nbor_sums[nbor.opinion] = [network[agent.unique_id][nbor.unique_id]['weight'], 1, [nbor.unique_id]]
            else:
                nbor_sums[nbor.opinion][0] += network[agent.unique_id][nbor.unique_id]['weight']
                nbor_sums[nbor.opinion][1] += 1
                nbor_sums[nbor.opinion][2].append(nbor.unique_id)

        most_common = max(nbor_sums, key=lambda k: nbor_sums[k][0])
        new_opinions.append(most_common)

    for i, agent in enumerate(model.schedule.agents):
        curr_opinion = agent.opinion
        # can only change opinion if the new opinion is actually different
        if curr_opinion != new_opinions[i] and model.get_random_weight(polarization) < threshold:  # check polarization
            agent.opinion = new_opinions[i]