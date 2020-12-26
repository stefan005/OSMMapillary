import argparse
import json
import os
import shutil
import time
import urllib
from pathlib import Path

import overpy
import pandas as pd

BASE_DIR = './ressources/mapillary_raw/'
# See https://www.mapillary.com/developer/api-documentation/

# TODO read from config.yaml
MAPILLARY_API_IM_SEARCH_URL = 'https://a.mapillary.com/v3/images?'
MAPILLARY_API_IM_RETRIEVE_URL = 'https://d1cuyjsrcm0gby.cloudfront.net/'
CLIENT_ID = 'TG1sUUxGQlBiYWx2V05NM0pQNUVMQTo2NTU3NTBiNTk1NzM1Y2U2'
nrOfImageDownloadsPerNode = 100
maxDistance = 5
min_quality_score = 2


'''
Script to download images using the Mapillary image search API.

Downloads images inside a rect (min_lat, max_lat, min_lon, max_lon).
'''


def create_dirs(base_path) -> None:
    Path(base_path).mkdir(parents=True, exist_ok=True)


def query_search_api(lon, lat, max_results) -> str:
    '''
    Send query to the search API and get dict with image data.
    '''

    # Create URL
    params = urllib.parse.urlencode(list(zip(
        ['client_id', 'closeto', 'radius', 'per_page', 'pano','min_quality_score'],
        [CLIENT_ID, ','.join([str(lon), str(lat)]), str(maxDistance), str(max_results), str('false'),str(min_quality_score)])), doseq=True)

    print(MAPILLARY_API_IM_SEARCH_URL + params)

    # Get data from server, then parse JSON
    query = urllib.request.urlopen(MAPILLARY_API_IM_SEARCH_URL + params).read()
    query = json.loads(query)['features']

    print("Result: {0} images in area.".format(len(query)))
    return query


def download_images(query, path, size=640) -> list:
    '''
    Download images in query result to path.

    Return list of downloaded images with lat,lon.
    There are four sizes available: 320, 640, 1024 (default), or 2048.
    '''
    im_size = "thumb-{0}.jpg".format(size)
    im_list = []

    for im in query:
        # Use key to create url to download from and filename to save into
        key = im['properties']['key']
        url = MAPILLARY_API_IM_RETRIEVE_URL + key + '/' + im_size
        filename = key + ".jpg"

        try:
            # Get image and save to disk
            if(os.path.isfile(path+filename) == False):
                image = urllib.request.urlretrieve(url, path + filename)
                #image.retrieve(url, path + filename)

                # Log filename and GPS location
                coords = ",".join(map(str, im['geometry']['coordinates']))
                im_list.append([filename, coords])

                print("Successfully downloaded: {0}".format(filename))
            else:
                print("file "+filename+" already exists")
        except KeyboardInterrupt:
            break
        except:
            print("Failed to download: {0}".format(filename))
    return im_list


def download_images_nearby(node, dirColumn, baseDir) -> None:
    path = baseDir + str(node[dirColumn]) + "/"
    create_dirs(path)
    query = query_search_api(
        node['nodeLon'], node['nodeLat'], nrOfImageDownloadsPerNode)
    download_images(query, path)


def get_labeled_data() -> None:
    # create directories for saving
    create_dirs(BASE_DIR + "labeled/")
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 49.729140, 9.857140, 49.822259, 10.008888 # Würzburg
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 48.629630, 9.386075, 48.848225, 9.014912 # Stuttgart
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 48.445283739783264,9.169244707902863,48.5177325182498,9.255454552487208 #Reutlingen
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 47.5399288434551,9.287805036523991,47.58562233499498,9.38796223098575 #Romanshorn (CH)
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 47.45821049392114,10.779062586126997,47.489848282639,10.854406692718158 #Plansee
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 47.39842578284083,11.458676624509508,47.45061953158145,11.605909070948542 #Karwendel
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 47.7899375706767,12.561436523959514,47.84948093480779,12.758245326012684 #Siegsdorf(Bayern)
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 47.30256210546281,13.039457154551314,47.349930838509835,13.17087134038843 #Schwarzach im Pongau
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 46.919649950229825,10.045231801229534,46.96595148201669,10.082757091129224#Silvretta (AT)
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 47.23220677687661,8.676197576814388,47.31308502409897,8.797109894245295 # Zürichsee Süd
    bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 49.36379746541175,8.331433176685323,49.4027511391986,8.740882735389278 # Mannheim/Heidelberg1
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 49.4027511391986,8.331433176685323,49.4727511391986,8.740882735389278 # Mannheim/Heidelberg2
    #bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 49.4727511391986,8.331433176685323,49.5727511391986,8.740882735389278 # Mannheim/Heidelberg3
    
    api = overpy.Overpass()
    result = api.query(
        f'node["surface"]({bbox_minlat},{bbox_minlon},{bbox_maxlat},{bbox_maxlon});way["surface"]["highway"]({bbox_minlat},{bbox_minlon},{bbox_maxlat},{bbox_maxlon});(._;>;);out;')
    nodes_surface = list()
    for way in result.ways:
        if("surface" in way.tags):
            for node in way.nodes:
                nodes_surface.append(
                    [node.id, node.lat, node.lon, way.id, way.tags["surface"]])
    df_nodes_surface = pd.DataFrame(
        nodes_surface, columns=['nodeId', 'nodeLat', 'nodeLon', 'wayId', 'waySurface'])
    df_nodes_surface = df_nodes_surface.loc[(df_nodes_surface.waySurface != "asphalt") | (df_nodes_surface.index%40 == 0)]
    df_nodes_surface.apply(download_images_nearby,
                           dirColumn='waySurface', baseDir=BASE_DIR + 'labeled/', axis=1)


def get_unlabeld_data() -> None:
    create_dirs(BASE_DIR + "unlabeled/")
    bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 49.729140, 9.857140, 49.822259, 10.008888

    api = overpy.Overpass()
    result = api.query(
        f'node({bbox_minlat},{bbox_minlon},{bbox_maxlat},{bbox_maxlon});way["highway"]({bbox_minlat},{bbox_minlon},{bbox_maxlat},{bbox_maxlon});(._;>;);out;')
    nodes_surface = list()
    for way in result.ways:
        if("surface" not in way.tags):
            for node in way.nodes:
                nodes_surface.append([node.id, node.lat, node.lon, way.id])
    df_nodes_wo_surface = pd.DataFrame(
        nodes_surface, columns=['nodeId', 'nodeLat', 'nodeLon', 'wayId'])
    df_nodes_wo_surface.apply(
        download_images_nearby, dirColumn='wayId', baseDir=BASE_DIR + 'unlabeled/', axis=1)


def main() -> None:
    get_labeled_data()
    get_unlabeld_data


if __name__ == "__main__":
    main()
