from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
import numpy as np

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    Household agent class with the following attribute:
    - opinion: nominal/discrete opinion (in this case between -1, 0, 1)
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model

        self.is_adapted = False  # Initial adaptation status set to False
        self.opinion = int(np.random.choice([-1, 0, 1]))
        self.loss_tolerance = 50000 * self.opinion  # Initial loss tolerance
        self.loss = 0

        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)

        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0

        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        self.flood_depth_actual = 0
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)

    def step(self):
        self.loss_tolerance = self.opinion * 50000

        if self.is_adapted:
            self.loss = self.model.cost_of_adaptation - self.flood_damage_estimated
        else:
            self.loss = self.flood_damage_estimated * 100000 - self.model.cost_of_adaptation
