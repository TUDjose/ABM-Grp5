import random

import mesa
import networkx as nx
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import numpy as np

# Import the agent class(es) from agents.py
from agents import Households

# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self,
                 seed=None,
                 number_of_households=25,
                 probability_of_network_connection=0.4,
                 number_of_nearest_neighbours=5,
                 polarization=0.5,  # 0 - 1
                 cost_of_adaptation=50000,
                 max_neighbors=5
                 ):

        super().__init__(seed=seed)
        self.seed = seed

        # variables for network
        self.number_of_households = number_of_households
        self.polarization = polarization
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_nearest_neighbours = number_of_nearest_neighbours
        self.max_neighbors = max_neighbors

        # model variables
        self.cost_of_adaptation = cost_of_adaptation

        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()

        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps('harvey')

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            # household.opinion = int(np.random.choice([-1, 0, 1]))   # set initial opinion
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)

        # Data collection setup to collect data
        model_metrics = {
            "adapted": self.total_adapted_households,
            "positive": self.total_positive_opinions,
            "negative": self.total_negative_opinions,
            "neutral": self.total_neutral_opinions,
            "losers": lambda m: sum([1 for agent in self.schedule.agents if agent.loss > 0]),
            "winners": lambda m: sum([1 for agent in self.schedule.agents if agent.loss < 0])
        }
        agent_metrics = {
            "FloodDamageEstimated": lambda a: a.flood_damage_estimated * 100000,
            "IsAdapted": "is_adapted",
            "opinion": "opinion",
            "loss": "loss"
        }
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

        # revise network
        self.apply_polarization()
        self.revise_neutral_weights()

    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'model/base_model_mesa/input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'model/base_model_mesa/input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'model/base_model_mesa/input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        """Return the total number of households that have adapted."""
        # BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        adapted_count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.is_adapted])
        return adapted_count

    def total_positive_opinions(self):
        return sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.opinion == 1])

    def total_negative_opinions(self):
        return sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.opinion == -1])

    def total_neutral_opinions(self):
        return sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.opinion == 0])

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        c = ['yellow', 'blue', 'red']
        for agent in self.schedule.agents:
            color = c[agent.opinion]
            ax.scatter(agent.location.x, agent.location.y, color=color, s=10, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0, 1), ha='center',
                        fontsize=9)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: -1, Blue: +1, Yellow: 0")

        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def plot_network(self, labels=False, big=False):
        fig, ax = plt.subplots(figsize=(7, 7))
        pos = nx.spring_layout(self.G)

        c = ['yellow', 'blue', 'red']
        for agent in self.schedule.agents:
            ax.scatter(pos[agent.unique_id][0], pos[agent.unique_id][1], color=c[agent.opinion], s=50)
            ax.annotate(str(agent.unique_id), (pos[agent.unique_id][0], pos[agent.unique_id][1]), textcoords="offset points", xytext=(0,4),
                        ha='center', fontsize=9 if not big else 15)

        nx.draw_networkx_edges(self.G, pos, arrows=False, alpha=0.5)
        # nx.draw(self.G, pos, node_color=colors, with_labels=True)
        if labels:
            labels = nx.get_edge_attributes(self.G, 'weight')
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
            # edge_labels = nx.get_edge_attributes(self.G, 'weight')
            # edge_label_pos = {u: (pos[u] + pos[v]) / 2 for u, v in self.G.edges()}
            # nx.draw_networkx_edge_labels(self.G, pos=edge_label_pos, edge_labels=edge_labels, font_color='red', label_pos=0.5)

        ax.set_title(f"Social Network State at Step {self.schedule.steps}", fontsize=12)
        plt.show()

    def get_neighbors(self, agent):
        neighbors = [self.schedule.agents[edge[1]] for edge in self.G.out_edges(agent.unique_id)]
        return [nbor.unique_id for nbor in neighbors]

    def get_neighbors_from_id(self, agent_id):
        neighbors = [self.schedule.agents[edge[1]] for edge in self.G.out_edges(agent_id)]
        return [nbor.unique_id for nbor in neighbors]

    def initialize_network(self):
        """Initialize and return the social network directed graph based on minimum spanning tree of a Watts-Strogatz graph."""
        G = nx.watts_strogatz_graph(n=self.number_of_households,
                    k=self.number_of_nearest_neighbours,
                    p=self.probability_of_network_connection,
                    seed=self.seed)

        graph = nx.minimum_spanning_tree(G)
        weights = {edge: np.random.uniform(0,1) for edge in graph.edges}
        WSDG = nx.DiGraph()

        for edge, weight in weights.items():
            WSDG.add_edge(edge[0], edge[1], weight=weight)
            WSDG.add_edge(edge[1], edge[0], weight=weight)

        return WSDG

    def apply_polarization(self):
        """Create new edges to increase the interconnectivity of the network according to the polarization parameter."""
        edges = []  # edges that need to be created
        for agent in self.schedule.agents:
            neighbors = [self.schedule.agents[edge[1]] for edge in self.G.out_edges(agent.unique_id)]
            if len(neighbors) < self.max_neighbors:  # only create new edges when agents have less than max_neighbors
                for other in self.schedule.agents:
                    if other != agent and other not in neighbors and any(
                            n for n in self.get_neighbors(other) if n in self.get_neighbors(agent)):
                        edges.append((agent.unique_id, other.unique_id))

        # prune edge list to remove duplicates
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                edges.remove((edge[1], edge[0]))

        for edge in edges:
            if np.random.uniform(0, 1) < self.polarization:  # add edge between agent and 2nd degree neighbor (other)
                self.G.add_edge(edge[0], edge[1], weight=np.random.uniform(0, 1))
                self.G.add_edge(edge[1], edge[0], weight=np.random.uniform(0, 1))

            else:  # add edge to other random node not in neighbors and not corresponding to an existing edge
                random_node = random.choice([n for n in self.schedule.agents if n not in self.get_neighbors_from_id(edge[0])
                                             and n != self.schedule.agents[edge[0]] and (edge[0], n.unique_id) not in self.G.edges])
                self.G.add_edge(edge[0], random_node.unique_id, weight=np.random.uniform(0, 1))
                self.G.add_edge(random_node.unique_id, edge[0], weight=np.random.uniform(0, 1))

    def revise_neutral_weights(self):
        """Revise the weights of the edges from neutral agents to increase the influence of neutral agents."""
        for agent in self.schedule.agents:
            neighbors = [self.schedule.agents[edge[1]] for edge in self.G.out_edges(agent.unique_id)]
            for nbor in neighbors:
                if nbor.opinion == 0:
                    self.G[nbor.unique_id][agent.unique_id]['weight'] *= 2.5

    @staticmethod
    def nominal_opinions(model: mesa.Model, network: nx.DiGraph) -> None:
        """
        Method used to set the opinions of the agents in the model. Nominal opinions are set based on the equation:
            O[i, t+1] = Max(Sum(w[i,j]) for each O[j,t])
        where: O[i,t+1] is the opinion of agent i at time t+1
               w[i,j] is the weight of the edge between agent i and agent j
               O[j,t] is the opinion of neighboring agent j at time t
        O[i, t] is the opinion of agent i at time t, considered to be either -1, 0 or 1

        params:
            mesa_model: the mesa model with the agents containing a nominal/discrete opinion parameter
            network: directional graph connecting the agents in the model with weighted edges
        """

        new_opinions: list[int] = []
        for agent in model.schedule.agents:         # iterate over each agent in the model to calculate new opinion
            nbor_sums: dict[int, float] = {}            # dictionary to store the sum of weights for each opinion
            neighbors: list[Households] = [model.schedule.agents[edge[1]] for edge in network.out_edges(agent.unique_id)]
            for nbor in neighbors:
                if nbor_sums.get(nbor.opinion) is None:
                    nbor_sums[nbor.opinion] = network[nbor.unique_id][agent.unique_id]['weight']
                else:
                    nbor_sums[nbor.opinion] += network[nbor.unique_id][agent.unique_id]['weight']

            most_common = max(nbor_sums, key=lambda k: nbor_sums[k])            # calculate the most common opinion among neighbors
            new_opinions.append(most_common)

        for i, agent in enumerate(model.schedule.agents):
            agent.opinion = new_opinions[i]         # update the opinion of each agent after having calculated all new opinions


    def step(self):
        """At step 15 there will be a flood"""

        # adapt agents right before flood
        if self.schedule.steps == 14:
            for agent in self.schedule.agents:
                if agent.flood_damage_estimated * 100000 > agent.loss_tolerance + self.cost_of_adaptation:
                    agent.is_adapted = True

        # update opinions
        AdaptationModel.nominal_opinions(self, self.G)

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
