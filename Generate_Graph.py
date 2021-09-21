import networkx as nx
from networkx import DiGraph
import os
import pickle
from shapely import geometry
import matplotlib.pyplot as plt
from Bldg_fit_func import fit_bldg_features
from Block_Graph import BlockGraph
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import shapely.geometry as sg
import shapely.affinity as sa
from utils import included_angle


coord_cutoff = 1000.0
directP_dict = {(0,'north'),(1,'south'),(2,'east'),(3,'west')}
resolution = 0.3

if __name__ == "__main__":

    fp = 'D:\\Sat_road\\result_newtest\\testblk_cv_chicago_drive'
    if not os.path.exists(fp):
        os.mkdir(fp)

    with open(os.path.join(fp,'blk_bldg_4'), "rb") as poly_file:
        loaded_polygon = pickle.load(poly_file)

    with open(os.path.join(fp,'blk_road_4'), "rb") as poly_file:
        loaded_road = pickle.load(poly_file)

    print(len(loaded_polygon))

    x_rd, y_rd = loaded_road.exterior.xy
    minx_rd = np.amin(x_rd)
    miny_rd = np.amin(y_rd)
    x_rd_cut = x_rd - minx_rd
    y_rd_cut = y_rd - miny_rd
    # plt.plot(x_rd, y_rd)s

    G = BlockGraph(1, 1, 1) # initialize the block graph

    sum_angle = 0.0
    line_list = []

    for i in range(4):
        curr_road = sg.LineString([(x_rd_cut[i], y_rd_cut[i]), (x_rd_cut[i+1], y_rd_cut[i+1])])
        line_list.append(curr_road)
        if i != 3:
            tmp_angle = included_angle(x_rd_cut[i], y_rd_cut[i], x_rd_cut[i+1], y_rd_cut[i+1], x_rd_cut[i + 2], y_rd_cut[i + 2]) # the included angle of current road line with the next line
            sum_angle += tmp_angle
            G.add_obj_node(nodeID=i, posx=curr_road.centroid.x, posy=curr_road.centroid.y, includ_angle=tmp_angle)
        else:
            tmp_angle = 2 * np.pi - sum_angle
            G.add_obj_node(nodeID=i, posx=curr_road.centroid.x, posy=curr_road.centroid.y, includ_angle = tmp_angle)



    for i in range(len(loaded_polygon)):
        curr_poly = loaded_polygon[i]

        curr_poly_cut = sa.translate(curr_poly, -minx_rd, -miny_rd)
        x, y = curr_poly.exterior.xy
        minx = np.amin(x)
        miny = np.amin(y)
        maxx = np.amax(x)
        maxy = np.amax(y)

        x = x - minx
        y = y - miny

        print((maxx - minx), (maxy - miny))

        width = np.float(maxx - minx) / resolution # Width of pixel in 0.3m resolution
        height = np.float(maxy - miny) / resolution # Height of pixel in 0.3m resolution

        dpi = 400
        w_inch = width / np.float(dpi)
        h_inch = height / np.float(dpi)

        fig = plt.figure(figsize=(w_inch, h_inch), dpi=dpi)
        plt.fill(x, y)

        ax = fig.gca()
        ax.axis('off')
        fig.tight_layout(pad=0)

        # To remove the huge white borders
        ax.margins(0)

        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        canvas = FigureCanvas(fig)
        canvas.draw()
        img_as_string, (width, height) = canvas.print_to_buffer()
        as_rgba = np.frombuffer(img_as_string, dtype='uint8').reshape((height, width, 4))

        img = as_rgba[:, :, :3]

        curr_shape, iou, curr_height, curr_width, theta = fit_bldg_features(img)
        # print(curr_shape, iou, curr_height, curr_width, theta)

        curr_area = curr_poly.area
        curr_posx = curr_poly.centroid.x
        curr_posy = curr_poly.centroid.y

        curr_height = curr_height * resolution # define that length should be larger than width
        curr_width = curr_width * resolution

        curr_length = curr_width
        curr_shortlength = curr_height
        curr_azimuth = theta % np.pi # clockwise

        # if curr_height <= curr_width:
        #     curr_length = curr_width
        #     curr_shortlength = curr_height
        #     curr_azimuth = theta % np.pi # clockwise
        # else:
        #     curr_length = curr_height
        #     curr_shortlength = curr_width
        #     curr_azimuth = (theta + 0.5 * np.pi) % (np.pi)   # clockwise % pi or % 0.5pi?

        curr_posx = curr_posx - minx_rd
        curr_posy = curr_posy - miny_rd


        print(curr_shape, iou, curr_area, curr_posx, curr_posy, curr_length, curr_shortlength, curr_azimuth)
        G.add_obj_node(nodeID = i, azimuth=curr_azimuth, posx=curr_posx, posy=curr_posy,
                        bldg_shortlength=curr_shortlength, bldg_length=curr_length, bldg_shape=curr_shape, bldg_area=curr_area)



    ####### initialize all nodes
    for i in range(len(line_list)):
        for j in range(len(loaded_polygon)):
            distance = line_list[i].distance(loaded_polygon[j])





    # test_line1 = sg.LineString([(x_rd[0], y_rd[0]),(x_rd[1], y_rd[1])])
    # test_line2 = sg.LineString([(x_rd[1], y_rd[1]),(x_rd[2], y_rd[2])])
    # test_line3 = sg.LineString([(x_rd[2], y_rd[2]),(x_rd[3], y_rd[3])])
    # test_line4 = sg.LineString([(x_rd[3], y_rd[3]),(x_rd[4], y_rd[4])])


    #     print(test_line1.distance(curr_poly))
    #     print(test_line2.distance(curr_poly))
    #     print(test_line3.distance(curr_poly))
    #     print(test_line4.distance(curr_poly))
    #
    # print(G.nodes(data=True))


    # G = BlockGraph(1, 1, 1)
    # G.add_obj_node(0, 0.1, 20, 30)
    # print(G.nodes(data=True))

    # G.add_obj_node(1, 0.1, 20, 30)
    # G.add_obj_edge(0, 1, 0.0, 0.5, 0)
    # print(G.population)
    # nx.write_gpickle(G, "test.gpickle")
    # a = nx.read_gpickle('test.gpickle')


    # def generate_graph_from_footprint(self, node, node):
