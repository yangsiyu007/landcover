import json
import os

import fiona
import fiona.transform
import shapely
import shapely.geometry
import utm

from DataLoader import DataLoaderCustom, DataLoaderUSALayer, DataLoaderBasemap
from web_tool import ROOT_DIR

_DATASET_FN = "datasets.json"


def get_area_from_geometry(geom, src_crs="epsg:4326"):
    if geom["type"] == "Polygon":
        lon, lat = geom["coordinates"][0][0]
    elif geom["type"] == "MultiPolygon":
        lon, lat = geom["coordinates"][0][0][0]
    else:
        raise ValueError("Polygons and MultiPolygons only")

    zone_number = utm.latlon_to_zone_number(lat, lon)
    hemisphere = "+north" if lat > 0 else "+south"
    dest_crs = "+proj=utm +zone=%d %s +datum=WGS84 +units=m +no_defs" % (zone_number, hemisphere)
    projected_geom = fiona.transform.transform_geom(src_crs, dest_crs, geom)
    area = shapely.geometry.shape(
        projected_geom).area / 1000000.0  # we calculate the area in square meters then convert to square kilometers
    return area


def _load_geojson_as_list(fn):
    """Returns shapes in input geojson as a list of shapely shapes, a list of their areas in km^2 and the CRS.

    We calculate area here by re-projecting the shape into its local UTM zone, converting it to a shapely `shape`,
    then using the `.area` property.
    
    Args:
        fn: Path to a geojson file
    """
    shapes = []
    areas = []
    with fiona.open(fn) as f:
        src_crs = f.crs
        for row in f:
            geom = row["geometry"]

            area = get_area_from_geometry(geom, src_crs)
            areas.append(area)

            shape = shapely.geometry.shape(geom)
            shapes.append(shape)
    return shapes, areas, src_crs


def _load_dataset(dataset):
    # Step 1: load the shape layers
    shape_layers = {}
    if dataset["shapeLayers"] is not None:
        for shape_layer in dataset["shapeLayers"]:
            fn = os.path.join(ROOT_DIR, shape_layer["shapesFn"])
            if os.path.exists(fn):
                shapes, areas, crs = _load_geojson_as_list(fn)
                shape_layer["geoms"] = shapes
                shape_layer["areas"] = areas
                shape_layer["crs"] = crs[
                    "init"]  # TODO: will this break with fiona version; I think `.crs` will turn into a PyProj object
                shape_layers[shape_layer["name"]] = shape_layer
            else:
                print("WARNING: Cannot load dataset because shape layer at {} does not exist".format(fn))
                return False

    # Step 2: make sure the dataLayer exists
    if dataset["dataLayer"]["type"] == "CUSTOM":
        fn = os.path.join(ROOT_DIR, dataset["dataLayer"]["path"])
        if not os.path.exists(fn):
            print("WARNING: Cannot load dataset because data layer at {} does not exist".format(fn))
            return False

    # Step 3: setup the appropriate DatasetLoader
    if dataset["dataLayer"]["type"] == "CUSTOM":
        data_loader = DataLoaderCustom(data_fn=dataset["dataLayer"]["path"],
                                       shapes=shape_layers,
                                       padding=dataset["dataLayer"]["padding"])

    elif dataset["dataLayer"]["type"] == "USA_LAYER":
        data_loader = DataLoaderUSALayer(shapes=shape_layers,
                                         padding=dataset["dataLayer"]["padding"])

    elif dataset["dataLayer"]["type"] == "BASEMAP":
        data_loader = DataLoaderBasemap(data_url=dataset["dataLayer"]["path"],  # TODO should this be path or url?
                                        padding=dataset["dataLayer"]["padding"])
    else:
        print("WARNING: Cannot load dataset because no appropriate DatasetLoader found for data layer type {}".format(
            dataset["dataLayer"]["type"]))
        return False

    return {
        "data_loader": data_loader,
        "shape_layers": shape_layers,
    }


def load_datasets():
    dataset_json = json.load(open(os.path.join(ROOT_DIR, _DATASET_FN)))
    datasets = dict()

    for key, dataset in dataset_json.items():
        dataset_object = _load_dataset(dataset)

        if dataset_object is False:
            print("WARNING: files are missing, we will not be able to serve '%s' dataset" % (key))
        else:
            datasets[key] = dataset_object

    return datasets
