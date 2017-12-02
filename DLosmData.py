# coding: utf-8

import urllib.request
import xml.etree.ElementTree as ET

# ## Get Nodes from OSM
import pandas as pd


def get_surfacenode_dataframe(boundingbox):
    # berlin
    # boundingbox=[12.8389,52.3030,13.9348,52.7246]
    # charlottenburg
    # boundingbox = [13.2457, 52.5045, 13.3470, 52.5288]
    # westend
    boundingbox = [13.2521, 52.5123, 13.2820, 52.5237]
    # r= requests.get("http://overpass-api.de/api/map?bbox=13.2457,52.5045,13.3470,52.5288")

    osmapi = osmapi.OsmApi()
    points = osmapi.Map(boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[3])

    # berlin
    # boundingbox=[12.8389,52.3030,13.9348,52.7246]
    # charlottenburg
    # boundingbox = [13.2457, 52.5045, 13.3470, 52.5288]
    # westend
    # boundingbox = [13.2521, 52.5123, 13.2820, 52.5237]
    # r= requests.get("http://overpass-api.de/api/map?bbox=13.2457,52.5045,13.3470,52.5288")
    # TODO URL an boundingbox anpassen
    url = "https://overpass-api.de/api/interpreter?data=node%5B%22surface%22%5D%2852%2E494853137273%2C13%2E242473602295%2C52%2E538935532664%2C13%2E321695327759%29%3Bway%5B%22surface%22%5D%5B%22highway%22%5D%2852%2E494853137273%2C13%2E242473602295%2C52%2E538935532664%2C13%2E321695327759%29%3B%28%2E%5F%3B%3E%3B%29%3Bout%3B%0A"

    # get Data
    response = urllib.request.urlopen(url)
    xml = response.read()
    response.close()

    # parse Data
    root = ET.fromstring(xml)

    # create surface Dataframe
    nodes_surface = list()
    for way in root.findall('way'):
        for child in list(way):
            if (child.attrib.get('k') == 'surface'):
                # print(child.attrib.get('v'))
                for node in way.findall('nd'):
                    nodes_surface.append([child.attrib.get('v'), way.attrib.get('id'), node.attrib.get('ref')])

    df_surface_node = pd.DataFrame(nodes_surface, columns=['surface', 'wayId', 'nodeId'])

    # create list of node
    node_list = list()
    for way in root.findall('node'):
        node_list.append([way.attrib.get('id'), way.attrib.get('lat'), way.attrib.get('lon')])
    df_nodes = pd.DataFrame(node_list, columns=['nodeId', 'lat', 'lon'])

    return (df_surface_node.merge(df_nodes, on='nodeId', how='left'))
