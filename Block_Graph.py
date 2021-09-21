import networkx as nx
from networkx import DiGraph
import os
import pickle
from shapely import geometry
import matplotlib.pyplot as plt

# class Node():
#     def __init__(self, blockID, population=None, elevation=None, lr_images=None, regionID=None, azimuth = 0.0):
#         super(Node, self).__init__()



class BlockGraph(DiGraph):
    '''
    Class that represents a single graph node
    '''

    def __init__(self, blockID, population=None, elevation=None, lr_images=None, regionID=None, azimuth = 0.0):
        super(BlockGraph, self).__init__()
        self.blockID = blockID
        self.regionID = regionID
        self.population = population
        self.elevation = elevation
        self.lr_images = lr_images

    def add_obj_node(self, nodeID, posx, posy,
                 azimuth = None, includ_angle = None,
                 bldg_shortlength=None, bldg_length=None, bldg_shape=None, bldg_area=None,
                 bldg_type=None, bldg_height=None,
                 road_type=None, road_length=None):

        self.add_node(nodeID, posx = posx, posy = posy,
                      azimuth=azimuth, included_angle = includ_angle,
                      bldg_shape = bldg_shape, bldg_area = bldg_area, bldg_shortlength = bldg_shortlength, bldg_length = bldg_length,
                            bldg_type = bldg_type,  bldg_height = bldg_height,
                            road_type = road_type, road_length = road_length)

    def add_obj_edge(self, start_id, end_id, edge_dist, orientation, edge_type):

        self.add_edge(start_id, end_id,
                            edge_dist = edge_dist, orientation = orientation, edge_type = edge_type)







