import unittest
import networkx as nx
import numpy as np
from model import AdaptationModel


class TestModel(unittest.TestCase):
    def test_model_network_MST(self):
        """Test that the initial network is a minimum spanning tree"""
        model = AdaptationModel(number_of_households=10, polarization=1)
        G = model.initialize_network()
        G = G.to_undirected()
        with self.assertRaises(nx.exception.NetworkXNoCycle) as context:
            nx.find_cycle(G, orientation='ignore')
        self.assertTrue("No cycle found." in str(context.exception))

    def test_model_network_directed(self):
        """Test that the initial network is directed and that all initial edges have correct weights"""
        model = AdaptationModel(number_of_households=10, polarization=1)
        G = model.initialize_network()
        self.assertTrue(G.is_directed())
        np.testing.assert_array_equal(G.out_edges(), [edge[::-1] for edge in G.in_edges()])
        self.assertTrue(all([0 <= G.edges[edge]['weight'] <= 1 for edge in G.edges()]))

    def test_apply_polarization(self):
        """Test that edges are created based on similar neighbours and polarization"""
        for p in range(50):
            model = AdaptationModel(number_of_households=6, polarization=(p+1)/50, max_neighbors=5)
            if (any(len(model.get_neighbors(agent)) >= model.max_neighbors for agent in model.schedule.agents) and
                    not any(len(cycle) > 2 for cycle in nx.simple_cycles(model.G))):
                self.assertEqual(len(model.created_edges), 0)
            else:
                self.assertTrue(len(nx.find_cycle(model.G, orientation='ignore')) > 0)

    def test_revise_neutral_weights(self):
        """Test that the weights of edges between neutral agents are increased"""
        model = AdaptationModel(number_of_households=6, polarization=1)
        edges = list(model.G.edges())
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                edges.remove((edge[1], edge[0]))

        for edge in model.G.edges():
            w = np.around(np.random.uniform(0, 1), decimals=4)
            model.G.edges[edge]['weight'] = w
            model.G.edges[(edge[1], edge[0])]['weight'] = w

        initial_weights = list(set([model.G.edges[edge]['weight'] for edge in model.G.edges() if model.schedule.agents[edge[0]].opinion ==
                                  0 or model.schedule.agents[edge[1]].opinion == 0]))

        model.revise_neutral_weights()
        final_weights = list(set([model.G.edges[edge]['weight'] for edge in model.G.edges() if model.schedule.agents[edge[0]].opinion ==
                                  0 or model.schedule.agents[edge[1]].opinion == 0]))

        for w in initial_weights:
            self.assertIn(w*2.5, final_weights)

    def test_nominal_opinions(self):
        """Test the nominal opinion function on a test case of the network"""
        model = AdaptationModel(number_of_households=6, polarization=1)
        # create test case
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2, 3, 4, 5])
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2,3), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (3, 2)])
        for edge in G.edges():
            G.edges[edge]['weight'] = 1
        model.G = G
        opinions = [-1, 0, 0, 0, 1, -1]
        for agent in model.schedule.agents:
            agent.opinion = opinions[agent.unique_id]
        model.revise_neutral_weights()
        AdaptationModel.nominal_opinions(model, model.G)
        new_opinions = [0, -1, 0, 0, -1, -1]
        for agent in model.schedule.agents:
            self.assertEqual(agent.opinion, new_opinions[agent.unique_id])

    def test_get_neighbours(self):
        model = AdaptationModel(number_of_households=6, polarization=1)
        # create test case
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2, 3, 4, 5])
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 3), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (3, 2)])
        for edge in G.edges():
            G.edges[edge]['weight'] = 1
        model.G = G
        neighbours = [[1, 2, 3, 4, 5], [0], [0, 3], [0, 2], [0], [0]]
        for agent in model.schedule.agents:
            np.testing.assert_array_equal(np.sort(model.get_neighbors(agent)), neighbours[agent.unique_id])