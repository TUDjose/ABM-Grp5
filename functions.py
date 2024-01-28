"""
@author: thoridwagenblast

Functions that are used in the model_file.py and agent.py for the running of the Flood Adaptation Model.
Functions get called by the Model and Agent class.
"""
import random
import numpy as np
import math
from shapely import contains_xy
from shapely import prepare
import geopandas as gpd
import networkx as nx


def set_initial_values(input_data, parameter, seed):
    """
    Function to set the values based on the distribution shown in the input data for each parameter.
    The input data contains which percentage of households has a certain initial value.

    Parameters
    ----------
    input_data: the dataframe containing the distribution of paramters
    parameter: parameter name that is to be set
    seed: agent's seed

    Returns
    -------
    parameter_set: the value that is set for a certain agent for the specified parameter
    """
    parameter_set = 0
    parameter_data = input_data.loc[(input_data.parameter == parameter)]  # get the distribution of values for the specified parameter
    parameter_data = parameter_data.reset_index()
    random.seed(seed)
    random_parameter = random.randint(0, 100)
    for i in range(len(parameter_data)):
        if i == 0:
            if random_parameter < parameter_data['value_for_input'][i]:
                parameter_set = parameter_data['value'][i]
                break
        else:
            if (random_parameter >= parameter_data['value_for_input'][i - 1]) and (
                    random_parameter <= parameter_data['value_for_input'][i]):
                parameter_set = parameter_data['value'][i]
                break
            else:
                continue
    return parameter_set


def get_flood_map_data(flood_map):
    """
    Getting the flood map characteristics.

    Parameters
    ----------
    flood_map: flood map in tif format

    Returns
    -------
    band, bound_l, bound_r, bound_t, bound_b: characteristics of the tif-file
    """
    band = flood_map.read(1)
    bound_l = flood_map.bounds.left
    bound_r = flood_map.bounds.right
    bound_t = flood_map.bounds.top
    bound_b = flood_map.bounds.bottom
    return band, bound_l, bound_r, bound_t, bound_b


shapefile_path = r'model/base_model_mesa/input_data/model_domain/houston_model/houston_model.shp'
floodplain_path = r'model/base_model_mesa/input_data/floodplain/floodplain_area.shp'

# Model area setup
map_domain_gdf = gpd.GeoDataFrame.from_file(shapefile_path)
map_domain_gdf = map_domain_gdf.to_crs(epsg=26915)
map_domain_geoseries = map_domain_gdf['geometry']
map_minx, map_miny, map_maxx, map_maxy = map_domain_geoseries.total_bounds
map_domain_polygon = map_domain_geoseries[0]  # The geoseries contains only one polygon
prepare(map_domain_polygon)

# Floodplain setup
floodplain_gdf = gpd.GeoDataFrame.from_file(floodplain_path)
floodplain_gdf = floodplain_gdf.to_crs(epsg=26915)
floodplain_geoseries = floodplain_gdf['geometry']
floodplain_multipolygon = floodplain_geoseries[0]  # The geoseries contains only one multipolygon
prepare(floodplain_multipolygon)


def generate_random_location_within_map_domain():
    """
    Generate random location coordinates within the map domain polygon.

    Returns
    -------
    x, y: lists of location coordinates, longitude and latitude
    """
    while True:
        # generate random location coordinates within square area of map domain
        x = random.uniform(map_minx, map_maxx)
        y = random.uniform(map_miny, map_maxy)
        # check if the point is within the polygon, if so, return the coordinates
        if contains_xy(map_domain_polygon, x, y):
            return x, y


def get_flood_depth(corresponding_map, location, band):
    """
    To get the flood depth of a specific location within the model domain.
    Households are placed randomly on the map, so the distribution does not follow reality.

    Parameters
    ----------
    corresponding_map: flood map used
    location: household location (a Shapely Point) on the map
    band: band from the flood map

    Returns
    -------
    depth: flood depth at the given location
    """
    row, col = corresponding_map.index(location.x, location.y)
    depth = band[row - 1, col - 1]
    return depth


def get_position_flood(bound_l, bound_r, bound_t, bound_b, img, seed):
    """
    To generater the position on flood map for a household.
    Households are placed randomly on the map, so the distribution does not follow reality.

    Parameters
    ----------
    bound_l, bound_r, bound_t, bound_b, img: characteristics of the flood map data (.tif file)
    seed: seed to generate the location on the map

    Returns
    -------
    x, y: location on the map
    row, col: location within the tif-file
    """
    random.seed(seed)
    x = random.randint(round(bound_l, 0), round(bound_r, 0))
    y = random.randint(round(bound_b, 0), round(bound_t, 0))
    row, col = img.index(x, y)
    return x, y, row, col


def calculate_basic_flood_damage(flood_depth):
    """
    To get flood damage based on flood depth of household
    from de Moer, Huizinga (2017) with logarithmic regression over it.
    If flood depth > 6m, damage = 1.

    Parameters
    ----------
    flood_depth : flood depth as given by location within model domain

    Returns
    -------
    flood_damage : damage factor between 0 and 1
    """
    if flood_depth >= 6:
        flood_damage = 1
    elif flood_depth < 0.025:
        flood_damage = 0
    else:
        # see flood_damage.xlsx for function generation
        flood_damage = 0.1746 * math.log(flood_depth) + 0.6483
    return flood_damage

def WSG(n, k, p, seed=None):
    """Returns a Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    newman_watts_strogatz_graph
    connected_watts_strogatz_graph

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.

    In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
    does not increase the number of edges. The rewired graph is not guaranteed
    to be connected as in :func:`connected_watts_strogatz_graph`.

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.Graph()
    nodes = list(range(n))  # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G
