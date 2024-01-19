import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from matplotlib.transforms import Affine2D
import random

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
                 flood_map_choice='harvey',
                 network='watts_strogatz',
                 probability_of_network_connection=0.4,
                 number_of_edges=3,
                 number_of_nearest_neighbours=5,
                 polarization=0.5,  # 0 - 1 (exclusive)
                 cost_of_adaptation=30000,
                 friend_radius=1,
                 p_seed=0
                 ):

        super().__init__(seed=seed)

        # model parameters
        self.polarization = polarization
        self.cost_of_adaptation = cost_of_adaptation
        self.friend_radius = friend_radius

        # defining the variables and setting the values
        self.number_of_households = number_of_households  # Total number of household agents
        self.seed = seed
        self.p_seed = p_seed

        # network
        self.network = network  # Type of network to be created
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours

        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network(type='rev_normal')
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)

        # Data collection setup to collect data
        model_metrics = {
            "total_adapted_households": self.total_adapted_households,
            "total_positive_opinions": self.total_opinions,
            "total_negative_opinions": lambda m: self.number_of_households - self.total_opinions()
        }
        agent_metrics = {
            "FloodDamageEstimated": "flood_damage_estimated",
            "IsAdapted": "is_adapted",
            "opinion": "opinion",
            "loss": "loss"
        }
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    def get_random_weight(self, stdev):
        # np.random.seed(self.p_seed)
        stdev = 1 - stdev
        a, b = (0 - 0.5) / stdev, (1 - 0.5) / stdev
        x = np.linspace(0, 1, 1000)
        y = truncnorm.pdf(x, a, b, loc=0.5, scale=stdev)
        reflection_transform = Affine2D().scale(1, -1).translate(0, np.max(y) + np.min(y))
        reflected_line = reflection_transform.transform_affine(np.column_stack((x, y)))
        w = np.random.choice(reflected_line[:, 0], p=reflected_line[:, 1] / np.sum(reflected_line[:, 1]))
        return np.around(w, 3)

    def random_weight(self, stdev):
        # np.random.seed(0)
        a, b = (0 - 0.5) / stdev, (1 - 0.5) / stdev
        return truncnorm.rvs(a, b, loc=0.5, scale=stdev)

    def initialize_network(self, type='uniform'):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        graph = nx.watts_strogatz_graph(n=self.number_of_households,
                                        k=self.number_of_nearest_neighbours,
                                        p=self.probability_of_network_connection,
                                        seed=self.seed)
        if type == 'uniform':
            np.random.seed(self.p_seed)
            for edge in graph.edges:
                graph.edges[edge]["weight"] = np.random.uniform(0, 1)
        elif type == 'rev_normal':
            for edge in graph.edges:
                graph.edges[edge]["weight"] = self.get_random_weight(self.polarization)

        return graph

    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'../input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'../input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'../input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
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

    def total_opinions(self):
        count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.opinion == 1])
        return count

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses
        for agent in self.schedule.agents:
            color = 'blue' if agent.is_adapted else 'red'
            ax.scatter(agent.location.x, agent.location.y, color=color, s=10, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0, 1), ha='center',
                        fontsize=9)
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue: adapted")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    @staticmethod
    def nominal_opinions(model, network, grid, radius):
        """
        Method used to set the opinions of the agents in the model. Nominal opinions are set based on the equation:
            O(i,t+1) = Sign(Sum(w(i,j) * O(j,t)))
        where: O(i,t+1) is the opinion of agent i at time t+1
               w(i,j) is the weight of the edge between agent i and agent j
               O(j,t) is the opinion of neighboring agent j at time t
        O(i, t) is the opinion of agent i at time t, considered ti be either -1 or 1

        params:
            mesa_model: the mesa model
            network: graph/network connecting the agents in the model with edges
            grid: mesa grid (created from the network) to find the neighbors of each agent
            radius: the radius of the network (number of edges away) to be considered as the neighborhood of an agent
        """
        # Iterate over all agents in the model
        for agent in model.schedule.agents:
            # find all (social) neighbors of the agent
            neighbors = grid.get_neighborhood(agent.pos, include_center=False, radius=radius)
            # get the weights of the edges between the agent and its neighbors
            weights = [network[agent.unique_id][neighbor]['weight'] for neighbor in neighbors]
            # get the opinions of the neighbors
            opinions = [model.schedule.agents[neighbor].opinion for neighbor in neighbors]
            # calculate the new opinion of the agent based on the equation above
            new_opinion = 1 if sum([weight * opinion for weight, opinion in zip(weights, opinions)]) >= 0 else -1
            # update the opinion of the current agent
            agent.opinion = new_opinion

    def step(self):
        """
        At step 7 there will be a flood
        """

        # adapt agents right before flood
        if self.schedule.steps == 7:
            for agent in self.schedule.agents:
                if agent.flood_damage_estimated * 100000 > agent.loss_tolerance + self.cost_of_adaptation:
                    agent.is_adapted = True

        # update opinions
        AdaptationModel.nominal_opinions(self, self.G, self.grid, self.friend_radius)

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
