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
min_quality_score = 4


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
    bbox_minlat, bbox_minlon, bbox_maxlat, bbox_maxlon = 49.729140, 9.857140, 49.822259, 10.008888 # Würzburg
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
    df_nodes_surface = df_nodes_surface.loc[(df_nodes_surface.waySurface != "asphalt") | (df_nodes_surface.index%25 == 0)]
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
