import json
# import laspy
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling
import time
import math
from sklearn.tree import export_graphviz
# from dtreeviz.trees import dtreeviz
# from IPython.display import Image
from sklearn import tree
from subprocess import check_call
from osgeo import ogr, osr, gdal
from skimage.filters import threshold_otsu
# from pyGEDI import *
import ATL08, general, regression, gee_fun
from arosics import COREG


def GEDI_L2A_download(bbox, version, out_dir, username="txx1220", password="Wsnd1220"):
    """
    bbox = [ul_lat,ul_lon,lr_lat,lr_lon]
    """
    session = sessionNASA(username, password)
    product_2A = 'GEDI02_A'
    gediDownload(out_dir, product_2A, version, bbox, session)


def json_merge(indir, key, outjson):
    """merge multiple json file into one json file
    indir: directory of the multiple json file
    key: key search in file name
    outjson: output location
    """
    gedijsonfiles = [indir + "\\" + f for f in os.listdir(indir) if f.endswith('.json') and key in f]
    data = []
    for g in gedijsonfiles:
        print(g)
        # for f in glob.glob(g):
        with open(g, 'r') as infile:
            sourcedata = json.load(infile)
            features = sourcedata['features']
            data.extend(features)

    outpath = indir + '\\' + outjson
    print(outpath)
    # data=json.dumps(data)
    with open(outpath, 'w') as outfile:
        json.dump({"features": data}, outfile)


def json_load(indir, injson, keys):
    with open(indir + injson, 'r') as f:
        source = json.load(f)

    features = source['features']
    data = []
    for feature in features:
        tmp = []
        for key in keys:
            tmp.append(feature['properties'][key])
        data.append(tmp)
    return np.asarray(data, dtype=float)


def qua_filter(indir, injson, outjson_good, outjson_bad, sensitivity):
    """
    Filter data according to the sensitivity feature
    :param indir: directory of the input json file
    :param injson: file name of the input json file
    :param outjson_good: remained good quality data
    :param outjson_bad: filtered bad quality data
    :param sensitivity: sensitivity level
    """
    good_data = []
    bad_data = []
    with open(indir + '\\' + injson, 'r') as f:
        source = json.load(f)
    features = source['features']
    print('Before filtering:' + str(len(features)))
    for feature in features:
        if feature['properties']['quality_flag'] != 0 and feature['properties']['sensitivity'] >= sensitivity:
            good_data.append(feature)
        else:
            bad_data.append(feature)
    with open(indir + '\\' + outjson_good, 'w') as f:
        json.dump({"features": good_data}, f)
    print('Good data left:' + str(len(good_data)))
    print('Bad data filtered:' + str(len(bad_data)))
    with open(indir + '\\' + outjson_bad, 'w') as f:
        json.dump({"features": bad_data}, f)
    return good_data, bad_data


def json2shp(indir, injson, desfile, epsg):
    """
    Convert json file to shapefile
    :param indir: directory of the input json file
    :param injson: file name of the input json file
    :param desfile: output shapefile
    :param epsg: coordinate reference system code
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(epsg)
    outputShapefile = indir + '\\' + desfile
    if os.path.exists(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer("basemap", crs, geom_type=ogr.wkbPoint)
    field_name = ogr.FieldDefn("elev", ogr.OFTReal)
    outLayer.CreateField(field_name)
    outLayer.CreateField(ogr.FieldDefn("mSeaLevel", ogr.OFTReal))

    with open(indir + '\\' + injson, 'r') as f:
        data = json.load(f)
    print(type(data))
    features = data['features']
    for i in features:
        feature = ogr.Feature(outLayer.GetLayerDefn())
        feature.SetField("elev", i['properties']['elev_lowestmode'])
        feature.SetField("mSeaLevel", i['properties']['mean_sea_surface'])
        wkt = "POINT(%f %f)" % (float(i['properties']['Longitude']), float(i['properties']['Latitude']))
        point = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(point)
        outLayer.CreateFeature(feature)
        feature = None
    outDataSet = None


def shp_projection(indir, inshp, outshp, tarepsg):
    """
    Projection of shapefile
    :param indir: direction of the input shapefile
    :param inshp: file name of the input shapefile
    :param outshp: output shapefile
    :param tarepsg: targeting coordinate reference system code
    :return:
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # get the input layer
    inDataSet = driver.Open(indir + '\\' + inshp)
    inLayer = inDataSet.GetLayer()
    # input SpatialReference
    inSpatialRef = inLayer.GetSpatialRef()
    # output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(tarepsg)
    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # create the output layer
    outputShapefile = indir + '\\' + outshp
    if os.path.exists(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer("basemap_6345", outSpatialRef, geom_type=ogr.wkbPoint)
    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)
    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()
    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()
    # Save and close the shapefiles
    inDataSet = None
    outDataSet = None


def rasterize(indir, vector_fn, raster_fn, pixel_size, field_name):
    """
    rasterize the vector shapefile into raster shapefile
    :param indir: directory of the input vector file
    :param vector_fn: file name of the input vector file
    :param raster_fn: output raster file
    :param pixel_size: cell size of target raster file
    :param field_name: file name of raster file
    :return:
    """
    NoData_value = -9999
    source_ds = ogr.Open(indir + '\\' + vector_fn)
    source_layer = source_ds.GetLayer(0)
    source_crs = source_layer.GetSpatialRef()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    print(x_min, x_max, y_min, y_max)
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('GTiff').Create(indir + '\\' + raster_fn, x_res, y_res, 1, gdal.GDT_Float32)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    target_ds.SetProjection(source_crs.ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    # Rasterize
    if field_name:
        gdal.RasterizeLayer(target_ds, [1], source_layer, options=["ATTRIBUTE={0}".format(field_name)])
    else:
        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[-1])


def elev_diff_dem(inshp, reftif, reftif_unit, outcsv):
    """
    Calculate the elevation difference between GEDI and reference DEM
    :param inshp: directory of the input shapefile (GEDI)
    :param reftif: directory of the reference DEM
    :param reftif_unit: unit of the elevation of reference data
    :param outcsv: output csv file
    """
    in_ds = ogr.Open(inshp)
    in_layer = in_ds.GetLayer(0)
    inFeatureCount = in_layer.GetFeatureCount()
    print('InLayer has {} features.'.format(inFeatureCount))
    in_crs = in_layer.GetSpatialRef()
    ref_tif = gdal.Open(reftif, 0)
    ref_band = ref_tif.GetRasterBand(1)
    print(ref_band)
    ref_crs = ref_tif.GetSpatialRef()
    # print(in_crs.ExportToWkt())
    # print(ref_crs.ExportToWkt())

    X = []
    Y = []
    Z_GEDI = []
    Z_3DEP = []
    MSL = []
    with rasterio.open(reftif) as src:
        refDEM = src.read(1)
        # print(refDEM)
        print("Start processing...")
        for feature in in_layer:
            geom = feature.GetGeometryRef()
            X.append(geom.GetX())
            Y.append(geom.GetY())
            Z_GEDI.append(feature.GetField('elev'))
            MSL.append(feature.GetField('mSeaLevel'))

            row, col = src.index(geom.GetX(), geom.GetY())
            Z_3DEP.append(refDEM[row, col])
    print("Finished processing...")
    # if the height of DEM is in feet:
    if reftif_unit == 'feet':
        Z_3DEP = np.asarray(Z_3DEP) * 0.3048
    # if the height of DEM is in feet:
    else:
        Z_3DEP = np.asarray(Z_3DEP)
    diff = np.asarray(Z_GEDI) - np.asarray(MSL) - Z_3DEP
    res = np.dstack((X, Y, Z_GEDI, Z_3DEP, diff, MSL))[0]
    # print("Start writing to csv...")
    np.savetxt(outcsv, res, delimiter=',', header="X,Y,Z_GEDI,Z_3DEP,diff,MSL_GEDI")
    print("Output saved as ", outcsv)


def z_interp(XY, las_file, kdtree, k):
    if len(XY.shape) == 1:
        n = 1
    else:
        n = XY.shape[0]
    num_returns = las_file.num_returns
    return_num = las_file.return_num
    z = lidarPC.scaled_dimension(las_file, 'z')
    z = z[num_returns == return_num]
    intensity = las_file.intensity
    intensity = intensity[num_returns == return_num]
    # geoDF.csr='EPSG:2968'
    dist, idx = kdtree.query(XY, k)
    dist = dist.reshape((n, k))
    idx = idx.reshape((n, k))
    wList = 1 / dist
    zList = z[idx]
    z_interp = np.sum(wList * zList, axis=1) / np.sum(wList, axis=1)
    z_nn = zList[:, 0]
    # output = np.asarray([z_interp, z_nn])
    return z_interp, z_nn


def get_Z_3dep(XY_gedi, las_dir, k):
    X = XY_gedi[:, 0]
    Y = XY_gedi[:, 1]
    las_kdtree_dir = las_dir + '\\kdtree'
    file_list_kdtree = [o for o in os.listdir(las_kdtree_dir) if o.endswith('.pickle')]
    file_list_las = [o for o in os.listdir(las_dir) if o.endswith('.las')]
    boundaries = np.loadtxt(las_dir + '\\boundaries.txt', delimiter=',')
    las_idx = -1 * np.ones(np.max(XY_gedi.shape))
    z = np.zeros(XY_gedi.shape)
    for i, bounds in enumerate(boundaries):
        start = time.time()
        if i % 10 == 0:
            print('Checking in file {}/{}'.format(i, len(file_list_las)))
        # tmp = np.bitwise_and(bounds[0] <= X, X <= bounds[2], las_idx == -1)
        # tmp = np.bitwise_and(tmp, bounds[1] <= Y, Y <= bounds[3])
        tmp = np.array(bounds[0] <= X) & np.array(X <= bounds[2]) & np.array(las_idx == -1) \
              & np.array(bounds[1] <= Y) & np.array(Y <= bounds[3])
        boolArray = np.where(tmp)[0]
        if boolArray.size == 0:
            continue
        else:
            las_idx[boolArray] = i
            with open(las_kdtree_dir + '\\' + file_list_kdtree[i], 'rb') as file:
                tree = pickle.load(file)
            las_file = laspy.read(las_dir + '\\' + file_list_las[i])
            z[boolArray, 0], z[boolArray, 1] = z_interp(XY_gedi[boolArray, :], las_file, tree, k)
            end = time.time()
            print('timer: ', end - start)
    return z


def load_GEDI_pars(indir, gedi_Z, gedi_lonlat, gedi_XY, atl08_XY, atl08_Z, distance_threshold, k):
    n_points = gedi_XY.shape[0]
    if os.path.exists(indir + "sentinel-2.csv"):
        df = pd.read_csv(indir + "sentinel-2.csv")
        NDVI = df['NDVI'].to_numpy()
        EVI = df['EVI'].to_numpy()
    else:
        gedi_tmp = np.concatenate(
            (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
            axis=-1)
        gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
        interval = 20000
        out_df = pd.DataFrame()
        for i in range(math.ceil(gedi_df.shape[0] / interval)):
            if i == math.ceil(gedi_df.shape[0] / interval) - 1:
                tmp = gedi_df[i * interval:]
            else:
                tmp = gedi_df[i * interval:(i + 1) * interval]
            tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'sentinel-2', ["2019-09-20", "2019-09-25"])
            out_df = out_df.append(tmp_out, ignore_index=True)
        out_df.to_csv(indir + "sentinel-2.csv")
        NDVI = out_df['NDVI'].to_numpy()
        EVI = out_df['EVI'].to_numpy()

    if os.path.exists(indir + "slope.csv"):
        df = pd.read_csv(indir + "slope.csv")
        slope = df['slope'].to_numpy()
    else:
        gedi_tmp = np.concatenate(
            (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
            axis=-1)
        gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
        interval = 20000
        out_df = pd.DataFrame()
        for i in range(math.ceil(gedi_df.shape[0] / interval)):
            if i == math.ceil(gedi_df.shape[0] / interval) - 1:
                tmp = gedi_df[i * interval:]
            else:
                tmp = gedi_df[i * interval:(i + 1) * interval]
            tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'slope', ["2019-09-20", "2019-09-25"])
            out_df = out_df.append(tmp_out, ignore_index=True)
        out_df.to_csv(indir + "slope.csv")
        slope = out_df['slope'].to_numpy()

    if os.path.exists(indir + "aspect.csv"):
        df = pd.read_csv(indir + "aspect.csv")
        aspect = df['aspect'].to_numpy()
    else:
        gedi_tmp = np.concatenate(
            (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
            axis=-1)
        gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
        interval = 20000
        out_df = pd.DataFrame()
        start = time.time()
        for i in range(math.ceil(gedi_df.shape[0] / interval)):
            if i == math.ceil(gedi_df.shape[0] / interval) - 1:
                tmp = gedi_df[i * interval:]
            else:
                tmp = gedi_df[i * interval:(i + 1) * interval]
            tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'aspect', ["2019-09-20", "2019-09-25"])
            out_df = out_df.append(tmp_out, ignore_index=True)
        out_df.to_csv(indir + "aspect.csv")
        print("extraction time:", (time.time() - start) / 60)
        aspect = out_df['aspect'].to_numpy()

    if os.path.exists(indir + "elevation.csv"):
        df = pd.read_csv(indir + "elevation.csv")
        elevation = df['elevation'].to_numpy()
    else:
        gedi_tmp = np.concatenate(
            (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
            axis=-1)
        gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
        interval = 20000
        out_df = pd.DataFrame()
        for i in range(math.ceil(gedi_df.shape[0] / interval)):
            if i == math.ceil(gedi_df.shape[0] / interval) - 1:
                tmp = gedi_df[i * interval:]
            else:
                tmp = gedi_df[i * interval:(i + 1) * interval]
            tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'elevation', ["2019-09-20", "2019-09-25"])
            out_df = out_df.append(tmp_out, ignore_index=True)
        out_df.to_csv(indir + "elevation.csv")
        elevation = out_df['elevation'].to_numpy()

    if os.path.exists(indir + "land_cover.csv"):
        df = pd.read_csv(indir + "land_cover.csv")
        lc = df['LC_Type1'].to_numpy()
    else:
        gedi_tmp = np.concatenate(
            (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
            axis=-1)
        gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
        interval = 20000
        out_df = pd.DataFrame()
        start = time.time()
        for i in range(math.ceil(gedi_df.shape[0] / interval)):
            if i == math.ceil(gedi_df.shape[0] / interval) - 1:
                tmp = gedi_df[i * interval:]
            else:
                tmp = gedi_df[i * interval:(i + 1) * interval]
            tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'LC_Type1', ["2019-09-20", "2019-09-25"])
            out_df = out_df.append(tmp_out, ignore_index=True)
        out_df.to_csv(indir + "land_cover.csv")
        end = time.time()
        print("Time for extraction of land cover: ", end - start)
        lc = out_df['LC_Type1'].to_numpy()

    if os.path.exists(indir + "interpolated_elip_height_from_ATL08_" + str(distance_threshold) + ".csv"):
        gedi_z_atl08_tmp = np.loadtxt(
            indir + "interpolated_elip_height_from_ATL08_" + str(distance_threshold) + ".csv", delimiter=',')
        gedi_z_atl08_interp = gedi_z_atl08_tmp[:, 1]
        gedi_valid_indices = gedi_z_atl08_tmp[:, 0].astype(np.int)
        # gedi_atl08_dist, gedi_atl08_idx = general.nn(gedi_XY, atl08_XY, k)
        # gedi_valid_indices1 = np.where(gedi_atl08_dist[:, 0] <= distance_threshold)[0]
        # gedi_valid_indices = np.intersect1d(gedi_valid_indices0, gedi_valid_indices1)
        # gedi_z_atl08_interp = gedi_z_atl08_interp[gedi_valid_indices]
    else:
        gedi_atl08_dist, gedi_atl08_idx = general.nn(gedi_XY, atl08_XY, k)
        gedi_valid_indices0 = np.where(gedi_atl08_dist[:, 0] <= distance_threshold)[0]

        print("finding the natural neighbors")
        gedi_z_atl08_interp = general.natural_neighbor_points(atl08_XY, atl08_Z, gedi_XY[gedi_valid_indices0, :])
        gedi_valid_indices1 = np.where(~np.isnan(gedi_z_atl08_interp))[0]
        gedi_valid_indices = gedi_valid_indices0[gedi_valid_indices1]
        gedi_z_atl08_interp = gedi_z_atl08_interp[gedi_valid_indices1]
        print("done finding the natural neighbors")
        # gedi_atl08_dist = gedi_atl08_dist[gedi_valid_indices, :]
        # gedi_atl08_idx = gedi_atl08_idx[gedi_valid_indices]
        #
        # gedi_z_atl08 = atl08_Z[gedi_atl08_idx]
        # gedi_z_atl08_interp = general.simple_idw(gedi_atl08_dist, gedi_z_atl08)
        # gedi_z_atl08_tmp = np.stack((gedi_valid_indices, gedi_z_atl08_interp), axis=-1)

        # gedi_z_atl08_interp = general.natural_neighbor_points(atl08_XY, atl08_Z, gedi_XY)
        # gedi_valid_indices1 = np.where(~np.isnan(gedi_z_atl08_interp))[0]
        # gedi_valid_indices = np.intersect1d(gedi_valid_indices0, gedi_valid_indices1)
        gedi_z_atl08_tmp = np.stack((gedi_valid_indices, gedi_z_atl08_interp), axis=-1)
        np.savetxt(indir + "interpolated_elip_height_from_ATL08_" + str(distance_threshold) + ".csv",
                   gedi_z_atl08_tmp, delimiter=',')

    NDVI = NDVI.reshape((NDVI.shape[0], 1))
    EVI = EVI.reshape((EVI.shape[0], 1))
    slope = slope.reshape((slope.shape[0], 1))
    aspect = aspect.reshape((aspect.shape[0], 1))
    land_cover = lc.reshape((lc.shape[0], 1))
    ortho_height_3dep1m = elevation.reshape((elevation.shape[0], 1))
    return NDVI, EVI, slope, aspect, land_cover, ortho_height_3dep1m, gedi_z_atl08_interp, gedi_valid_indices


def load_atl08_pars(indir, atl08_data, in_epsg, out_epsg):
    if os.path.exists(indir + 'geoid_height.csv'):
        atl08_geoid_height = np.loadtxt(indir + 'geoid_height.csv', delimiter=',')
    else:
        egm_path = "C:\\Users\\tian133\\OneDrive - purdue.edu\\Projects\\GEDI_ICESAT2\\data\\egm\\geoids\\egm2008-5.pgm"
        atl08_geoid_height = general.get_geoid_height(atl08_data[:, 0], atl08_data[:, 1], egm_path)
        np.savetxt(indir + 'geoid_height.csv', atl08_geoid_height, delimiter=',')
    atl08_XY = general.point_project(atl08_data[:, 0], atl08_data[:, 1], in_epsg, out_epsg)

    if os.path.exists(indir + 'ortho_height_from_3dep1m.csv'):
        ortho_height_3dep1m = np.loadtxt(indir + 'ortho_height_from_3dep1m.csv', delimiter=',')
    else:
        ortho_height_3dep1m = general.get_data_ee(atl08_data[:, 0:2], 'elevation')
        np.savetxt(indir + 'ortho_height_from_3dep1m.csv', ortho_height_3dep1m, delimiter=',')
    atl08_EH_3dep_1m = ortho_height_3dep1m + atl08_geoid_height
    return atl08_XY, atl08_EH_3dep_1m


def GEDI_analysis(indir, gedi_tmp, pars_list, analysis, cap_analysis, footprint_visual, cap_visual0, cap_visual1,
                  classify, minZ, maxZ):
    if analysis:
        general.data_analysis("GEDI", gedi_tmp, pars_list, plot=True,
                              save_path=indir + cap_analysis + ".png")
    if footprint_visual:
        gedi_ellipsoidH = gedi_tmp[:, 4]
        gedi_EH_3dep_1m = gedi_tmp[:, 5]
        gedi_height_error = (gedi_ellipsoidH - gedi_EH_3dep_1m)
        if classify:
            gedi_height_classified = np.copy(gedi_ellipsoidH)
            gedi_height_classified[gedi_height_classified >= maxZ] = math.ceil(maxZ) + 1
            gedi_height_classified[gedi_height_classified <= minZ] = math.ceil(minZ) - 1
            tmp = np.concatenate(
                (gedi_tmp[:, 0:2], gedi_height_classified.reshape((gedi_height_classified.shape[0], 1))),
                axis=1)

            gedi_height_error_classified = np.copy(gedi_height_error)
            gedi_height_error_classified[gedi_height_error_classified >= 20] = 21
            gedi_height_error_classified[gedi_height_error_classified <= -20] = -21
            tmp1 = np.concatenate((gedi_tmp[:, 0:2],
                                   gedi_height_error_classified.reshape((gedi_height_error_classified.shape[0], 1))),
                                  axis=1)
        else:
            tmp = gedi_tmp[:, [0, 1, 4]]
            tmp1 = np.concatenate((gedi_tmp[:, 0:2], gedi_height_error.reshape((gedi_height_error.shape[0], 1))),
                                  axis=1)
        general.footprint_visualization(tmp, ['lon', 'lat', 'height'], 2,
                                        caption=cap_visual0,
                                        save_path=indir + cap_visual0 + ".html")
        general.footprint_visualization(tmp1, ["lon", "lat", "height error (m)"], 2,
                                        caption=cap_visual1,
                                        save_path=indir + cap_visual1 + ".html")


def atl08_analysis(indir, atl08_tmp, pars_list, analysis, cap_analysis, footprint_visual, cap_visual0, cap_visual1):
    if analysis:
        general.data_analysis("ATL08", atl08_tmp, pars_list, plot=True,
                              save_path=indir + cap_analysis + ".png")
    if footprint_visual:
        general.footprint_visualization(atl08_tmp[:, [0, 1, 4]], ['lon', 'lat', 'height'], 2,
                                        caption=cap_visual0,
                                        save_path=indir + cap_visual0 + ".html")
        height_error = (atl08_tmp[:, 4] - atl08_tmp[:, 5])
        tmp = np.concatenate((atl08_tmp[:, :2], height_error.reshape((atl08_tmp.shape[0], 1))), axis=1)
        general.footprint_visualization(tmp, ["lon", "lat", "height error (m)"], 2,
                                        caption=cap_visual1,
                                        save_path=indir + cap_visual1 + ".html")


def tree_visual(RF_regressor, tree_index, features, labels, feature_list, save_path):
    # extract single tree
    tree0 = RF_regressor.estimators_[tree_index]
    viz = dtreeviz(tree0,
                   features,
                   labels,
                   target_name='height prediction',
                   orientation='LR',
                   feature_names=feature_list)
    viz.view()


def regress(gedi_dir, features, labels, feature_list, distance_threshold, regress_method):
    print('Total points:', labels.shape[0])
    path = gedi_dir + 'regression_model\\' + regress_method + '\\' + regress_method

    if os.path.exists(
            path + '_model_' + str(distance_threshold) + '.pickle'):
        with open(path + '_model_' + str(distance_threshold) + '.pickle', 'rb') as p:
            regressor = pickle.load(p)
        if "random_forest" in regress_method:
            with open(path + '_scaler_' + str(distance_threshold) + '.pickle', 'rb') as p:
                scaler = pickle.load(p)
        with open(path + '_error_test_feature_' + str(distance_threshold) + '.pickle', 'rb') as p:
            errors = pickle.load(p)
        with open(path + '_test_feature_' + str(distance_threshold) + '.pickle', 'rb') as p:
            test_features = pickle.load(p)
        with open(path + '_test_label_' + str(distance_threshold) + '.pickle', 'rb') as p:
            test_labels = pickle.load(p)
        with open(path + '_test_indices_' + str(distance_threshold) + '.pickle', 'rb') as p:
            test_indices = pickle.load(p)
        with open(path + '_train_feature_' + str(distance_threshold) + '.pickle', 'rb') as p:
            train_features = pickle.load(p)
        with open(path + '_train_label_' + str(distance_threshold) + '.pickle', 'rb') as p:
            train_labels = pickle.load(p)
        with open(path + '_train_indices_' + str(distance_threshold) + '.pickle', 'rb') as p:
            train_indices = pickle.load(p)
        with open(path + '_prediction_test_feature_' + str(distance_threshold) + '.pickle', 'rb') as p:
            prediction = pickle.load(p)
    else:
        if "random_forest" in regress_method:
            regressor, scaler, prediction, errors, train_features, test_features, \
            train_labels, test_labels, train_indices, test_indices = \
                regression.regressor_RF(features, labels, feature_list,
                                        save_path=gedi_dir + 'regression_model\\' + regress_method +
                                                  '\\variable importance ' +
                                                  str(distance_threshold) + '.png')
            with open(path + '_scaler_' + str(distance_threshold) + '.pickle', 'wb') as p:
                pickle.dump(scaler, p)
        elif regress_method == "spline":
            regressor, prediction, errors, train_features, test_features, \
            train_labels, test_labels, train_indices, test_indices = regression.regressor_spline(features, labels)
        elif regress_method == "svm":
            regressor, prediction, errors, train_features, test_features, \
            train_labels, test_labels, train_indices, test_indices = regression.regressor_svm(features, labels)
        elif regress_method == 'poly':
            regressor, prediction, errors, train_features, test_features, \
            train_labels, test_labels, train_indices, test_indices = regression.regressor_polyn(features, labels)
        elif regress_method == 'MLP':
            regressor, scaler, prediction, errors, train_features, test_features, \
            train_labels, test_labels, train_indices, test_indices = regression.regressor_MLP(features, labels)
            with open(path + '_scaler_' + str(distance_threshold) + '.pickle', 'wb') as p:
                pickle.dump(scaler, p)
        else:
            return None
        with open(path + '_model_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(regressor, p)
        with open(path + '_prediction_test_feature_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(prediction, p)
        with open(path + '_error_test_feature_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(errors, p)
        with open(path + '_test_feature_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(test_features, p)
        with open(path + '_test_label_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(test_labels, p)
        with open(path + '_test_indices_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(test_indices, p)
        with open(path + '_train_feature_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(train_features, p)
        with open(path + '_train_label_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(train_labels, p)
        with open(path + '_train_indices_' + str(distance_threshold) + '.pickle', 'wb') as p:
            pickle.dump(train_indices, p)
    if "random_forest" in regress_method:
        return regressor, scaler, prediction, errors, train_features, test_features, \
               train_labels, test_labels, train_indices, test_indices
    else:
        return regressor, prediction, errors, train_features, test_features, \
               train_labels, test_labels, train_indices, test_indices


def process(gedi_dir, gedi_keys, atl08_dir, atl08_keys, in_epsg, out_epsg, regress_method, distance_threshold,
            feature_list):
    gedi_df, _ = general.csv_load(gedi_dir, 'gedi', gedi_keys)
    gedi_data = gedi_df.to_numpy()
    gedi_data = gedi_data[~np.isnan(gedi_data).any(axis=1), :]
    num_good_quality = len(gedi_data[gedi_data[:, -1] >= 0.95, :])
    print('GEDI data with sensitivity smaller than 0.95: ', 1-num_good_quality/len(gedi_data))
    gedi_ellipsoidH = gedi_data[:, 2]
    threshold = threshold_otsu(gedi_ellipsoidH)
    gedi_data = gedi_data[(gedi_ellipsoidH <= threshold), :]
    gedi_ellipsoidH = gedi_ellipsoidH[(gedi_ellipsoidH <= threshold)]

    gedi_num_pts = gedi_data.shape[0]
    print('GEDI num of footprint. Raw data size: {}, filtered: {}.'.format(gedi_df.shape[0], gedi_num_pts))
    if os.path.exists(gedi_dir + 'gedi_XY.csv'):
        df = pd.read_csv(gedi_dir + 'gedi_XY.csv')
        gedi_XY = df.to_numpy()[:, 1:]
    else:
        gedi_XY = general.point_project(gedi_data[:, 0], gedi_data[:, 1], in_epsg, out_epsg)
        df = pd.DataFrame(gedi_XY, columns=['X', 'Y'])
        df.to_csv(gedi_dir + 'gedi_XY.csv')
    if os.path.exists(gedi_dir + 'gedi_geoidH.csv'):
        df = pd.read_csv(gedi_dir + 'gedi_geoidH.csv')
        gedi_geoidH = df.to_numpy()[:, 1]
        gedi_orthoH = gedi_ellipsoidH - gedi_geoidH
    else:
        gedi_geoidH = general.get_geoid_height(gedi_data[:, 0], gedi_data[:, 1])
        df = pd.DataFrame(gedi_geoidH, columns=['gedi_geoidH'])
        df.to_csv(gedi_dir + 'gedi_geoidH.csv')
        gedi_orthoH = gedi_ellipsoidH - gedi_geoidH
    if os.path.exists(gedi_dir + 'gedi_orthoH_srtm90.csv'):
        df = pd.read_csv(gedi_dir + 'gedi_orthoH_srtm90.csv')
        gedi_orthoH_srtm90 = df["orthometric_height"].to_numpy()
        gedi_ellipsoidH_srtm90 = gedi_orthoH_srtm90 + gedi_geoidH
    else:
        path_srtm = gedi_dir.replace('\\GEDI\\', '\\others\\') + 'srtm.tif'
        path_save = gedi_dir + 'gedi_orthoH_srtm90.csv'
        gedi_orthoH_srtm90 = general.extract_value_from_raster(gedi_data[:, :2], 'orthometric_height', path_srtm,
                                                               path_save)
        gedi_ellipsoidH_srtm90 = gedi_orthoH_srtm90 + gedi_geoidH
    # if os.path.exists(gedi_dir + 'gedi_ndvi.csv'):
    #     df = pd.read_csv(gedi_dir + 'gedi_ndvi.csv')
    #     gedi_ndvi = df["ndvi"].to_numpy()
    # else:
    #     path_ndvi = gedi_dir.replace('\\GEDI\\', '\\others\\') + 'ndvi.tif'
    #     path_save = gedi_dir + 'gedi_ndvi.csv'
    #     gedi_ndvi = general.extract_value_from_raster(gedi_data[:, :2], 'ndvi', path_ndvi,
    #                                                            path_save)
    if os.path.exists(gedi_dir + 'gedi_aspect.csv'):
        df = pd.read_csv(gedi_dir + 'gedi_aspect.csv')
        gedi_aspect = df["aspect"].to_numpy()
    else:
        path_aspect = gedi_dir.replace('\\GEDI\\', '\\others\\') + 'aspect.tif'
        path_save = gedi_dir + 'gedi_aspect.csv'
        gedi_aspect = general.extract_value_from_raster(gedi_data[:, :2], 'aspect', path_aspect,
                                                               path_save)
    if os.path.exists(gedi_dir + 'gedi_slope.csv'):
        df = pd.read_csv(gedi_dir + 'gedi_slope.csv')
        gedi_slope = df["slope"].to_numpy()
    else:
        path_slope = gedi_dir.replace('\\GEDI\\', '\\others\\') + 'slope.tif'
        path_save = gedi_dir + 'gedi_slope.csv'
        gedi_slope = general.extract_value_from_raster(gedi_data[:, :2], 'slope', path_slope,
                                                               path_save)

    if os.path.exists(gedi_dir + 'gedi_3dep.csv'):
        df = pd.read_csv(gedi_dir + 'gedi_3dep.csv')
        gedi_orthoH_3dep = df["elevation"].to_numpy()
        if "IN" in gedi_dir:
            gedi_orthoH_3dep = gedi_orthoH_3dep * 0.3048
    else:
        if "CA" in gedi_dir:
            path_3dep = "D:\\Project_ICESat-2\\data\\3DEP\\CA\\1-meter\\CA_1m3DEP.tif"
        else:
            path_3dep = "D:\\Project_ICESat-2\\data\\3DEP\\Tipp\\3DEP_TipC.tif"
        path_save = gedi_dir + 'gedi_3dep.csv'
        # XY = general.point_project(gedi_data[:, 0], gedi_data[:, 1], 4326, 26910)
        XY = general.point_project(gedi_data[:, 0], gedi_data[:, 1], 4326, 2968)
        gedi_orthoH_3dep = general.extract_value_from_raster(XY, 'elevation', path_3dep,
                                                               path_save)
        if "IN" in gedi_dir:
            gedi_orthoH_3dep = np.asarray(gedi_orthoH_3dep) * 0.3048

    # if os.path.exists(gedi_dir + 'gedi_evi.csv'):
    #     df = pd.read_csv(gedi_dir + 'gedi_evi.csv')
    #     gedi_evi = df["evi"].to_numpy()
    # else:
    #     path_evi = gedi_dir.replace('\\GEDI\\', '\\others\\') + 'evi.tif'
    #     path_save = gedi_dir + 'gedi_evi.csv'
    #     gedi_evi = general.extract_value_from_raster(gedi_data[:, :2], 'evi', path_evi,
    #                                                            path_save)

    gedi_lonlat = gedi_data[:, 0:2]

    atl08_df, _ = general.csv_load(atl08_dir+'\\rawdata\\', 'atl08', atl08_keys)
    atl08_data = atl08_df.to_numpy()
    atl08_ellipsoidH = atl08_data[:, 2]

    if os.path.exists(atl08_dir + 'atl08_XY.csv'):
        df = pd.read_csv(atl08_dir + 'atl08_XY.csv')
        atl08_XY = df.to_numpy()[:, 1:]
    else:
        atl08_XY = general.point_project(atl08_data[:, 0], atl08_data[:, 1], in_epsg, out_epsg)
        df = pd.DataFrame(atl08_XY, columns=['X', 'Y'])
        df.to_csv(atl08_dir + 'atl08_XY.csv')
    if os.path.exists(atl08_dir + 'atl08_geoidH.csv'):
        df = pd.read_csv(atl08_dir + 'atl08_geoidH.csv')
        atl08_geoidH = df.to_numpy()[:, 1]
        atl08_orthoH = atl08_ellipsoidH - atl08_geoidH
    else:
        atl08_geoidH = general.get_geoid_height(atl08_data[:, 0], atl08_data[:, 1])
        df = pd.DataFrame(atl08_geoidH, columns=['atl08_geoidH'])
        df.to_csv(atl08_dir + 'atl08_geoidH.csv')
        atl08_orthoH = atl08_ellipsoidH - atl08_geoidH
    if os.path.exists(atl08_dir + 'atl08_orthoH_srtm90.csv'):
        df = pd.read_csv(atl08_dir + 'atl08_orthoH_srtm90.csv')
        atl08_orthoH_srtm90 = df["orthometric_height"].to_numpy()
        # atl08_ellipsoidH_srtm90 = atl08_orthoH_srtm90 + atl08_geoidH
    else:
        path_srtm = gedi_dir.replace('\\GEDI\\', '\\others\\') + 'srtm.tif'
        path_save = atl08_dir + 'atl08_orthoH_srtm90.csv'
        atl08_orthoH_srtm90 = general.extract_value_from_raster(atl08_data[:, :2], 'orthometric_height', path_srtm,
                                                                path_save)
        # atl08_ellipsoidH_srtm90 = atl08_orthoH_srtm90 + atl08_geoidH
    if os.path.exists(atl08_dir + 'atl08_3dep.csv'):
        df = pd.read_csv(atl08_dir + 'atl08_3dep.csv')
        atl08_orthoH_3dep = df["elevation"].to_numpy()
        if "IN" in atl08_dir:
            atl08_orthoH_3dep = atl08_orthoH_3dep * 0.3048
    else:
        if "CA" in atl08_dir:
            path_3dep = "D:\\Project_ICESat-2\\data\\3DEP\\CA\\1-meter\\CA_1m3DEP.tif"
        else:
            path_3dep = "D:\\Project_ICESat-2\\data\\3DEP\\Tipp\\3DEP_TipC.tif"
        path_save = atl08_dir + 'atl08_3dep.csv'
        # XY = general.point_project(atl08_data[:, 0], atl08_data[:, 1], in_epsg, 26910)
        XY = general.point_project(atl08_data[:, 0], atl08_data[:, 1], in_epsg, 2968)
        atl08_orthoH_3dep = general.extract_value_from_raster(XY, 'elevation', path_3dep,
                                                               path_save)
        if "IN" in atl08_dir:
            atl08_orthoH_3dep = np.asarray(atl08_orthoH_3dep) * 0.3048

    filter_idx = general.filter(atl08_orthoH, atl08_orthoH_srtm90, [0.1, 0.9])
    atl08_data = atl08_data[filter_idx, :]
    atl08_al_slope = atl08_data[:, 3]
    atl08_orbit_orient = atl08_data[:, 4]
    atl08_XY = atl08_XY[filter_idx, :]
    atl08_orthoH_3dep = atl08_orthoH_3dep[filter_idx]
    atl08_orthoH = atl08_orthoH[filter_idx]
    atl08_orthoH_srtm90 = atl08_orthoH_srtm90[filter_idx]
    atl08_num_pts = atl08_data.shape[0]
    print('ATL08 num of footprint. Raw data size: {}, filtered: {}.'.format(atl08_df.shape[0], atl08_num_pts))

    if os.path.exists(gedi_dir + 'gedi_slope_atl08.csv'):
        df = pd.read_csv(gedi_dir + 'gedi_slope_atl08.csv', header=None)
        gedi_slopes = df.to_numpy()
    else:
        target = gedi_XY
        source_pts = np.concatenate((atl08_XY, atl08_orthoH.reshape((atl08_num_pts, 1)), atl08_data[:, 3:5]), axis=1)
        atl08_XY0 = atl08_XY[atl08_data[:, 4] == 0, :]
        atl08_data0 = atl08_data[atl08_data[:, 4] == 0, :]
        atl08_XY1 = atl08_XY[atl08_data[:, 4] == 1, :]
        atl08_data1 = atl08_data[atl08_data[:, 4] == 1, :]
        dist, idx = general.nn(gedi_XY, atl08_XY0, 20)
        gedi_along_slope0 = general.simple_idw(dist, atl08_data0[idx, 3])
        dist, idx = general.nn(gedi_XY, atl08_XY1, 20)
        gedi_along_slope1 = general.simple_idw(dist, atl08_data1[idx, 3])
        # gedi_along_slopes = general.natural_neighbor_points(atl08_XY, atl08_data[:, 3], gedi_XY)
        gedi_slopes = np.zeros((target.shape[0], 4))
        gedi_slopes[:, 0] = gedi_along_slope0
        gedi_slopes[:, 1] = gedi_along_slope1
        # gedi_slopes = ATL08.get_slopes(target, source_pts)
        path_save = gedi_dir + 'gedi_slope_atl08.csv'
        np.savetxt(path_save, gedi_slopes, delimiter=',')

    gedi_tmp = np.concatenate(
        (gedi_lonlat, gedi_XY, gedi_orthoH.reshape((gedi_num_pts, 1)), gedi_orthoH_3dep.reshape((gedi_num_pts, 1))),
        axis=1)
    GEDI_analysis(gedi_dir, gedi_tmp, ["lon", "lat", "X", "Y", "height", "reference height"],
                  analysis=False, cap_analysis="GEDI orginal footprint analysis",
                  footprint_visual=False, cap_visual0="GEDI Footprint in Tippecanoe County",
                  cap_visual1="Height error of GEDI footprint w.r.t 3DEP DEM", classify=True, minZ=110, maxZ=270)

    atl08_tmp = np.concatenate(
        (atl08_data[:, :2], atl08_XY, atl08_orthoH.reshape((atl08_num_pts, 1)),
         atl08_orthoH_3dep.reshape((atl08_num_pts, 1))), axis=1)
    atl08_analysis(atl08_dir, atl08_tmp, ["lon", "lat", "X", "Y", "height", "reference height"],
                   analysis=False, cap_analysis="atl08 all footprint analysis",
                   footprint_visual=False, cap_visual0="ATL08 Footprint in Tippecanoe County",
                   cap_visual1="Height error of ATL08 footprint w.r.t 3DEP DEM")

    # feature_list = ['X', 'Y', 'GEDI original height', 'degrade flag', 'quality flag', 'sensitivity',
    #                 'ndvi', 'evi', 'slope', 'aspect', 'land cover']
    # features_all = np.concatenate((gedi_XY, gedi_data[:, 2:-1]), axis=1)
    # feature_list = ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity', 'slope', 'aspect']
    # features_all = np.stack((gedi_orthoH, gedi_data[:, 3], gedi_data[:, 4], gedi_data[:, 5], gedi_slope, gedi_aspect),
    #                         axis=-1)
    if feature_list == ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity',
                        'slope1', 'slope2', 'slope3', 'slope4']:
        features_all = np.stack((gedi_orthoH, gedi_data[:, 3], gedi_data[:, 4], gedi_data[:, 5],
                                 gedi_slopes[:, 0], gedi_slopes[:, 1], gedi_slopes[:, 2], gedi_slopes[:, 3]), axis=-1)
    elif feature_list == ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity']:
        features_all = np.stack((gedi_orthoH, gedi_data[:, 3], gedi_data[:, 4], gedi_data[:, 5]), axis=-1)
    elif feature_list == ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity', 'slope', 'aspect']:
        features_all = np.stack((gedi_orthoH, gedi_data[:, 3], gedi_data[:, 4], gedi_data[:, 5],
                                 gedi_slope, gedi_aspect), axis=-1)
    elif feature_list == ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity',
                        'slope1', 'slope2']:
        features_all = np.stack((gedi_orthoH, gedi_data[:, 3], gedi_data[:, 4], gedi_data[:, 5],
                                 gedi_slopes[:, 0], gedi_slopes[:, 1]), axis=-1)
    else:
        print("Invalid feature list, try again.")
        return

    gedi_orthoH_srtm90 = gedi_orthoH_srtm90[~np.isnan(features_all).any(axis=1)]
    gedi_orthoH_3dep = gedi_orthoH_3dep[~np.isnan(features_all).any(axis=1)]
    gedi_XY = gedi_XY[~np.isnan(features_all).any(axis=1)]
    gedi_tmp = gedi_tmp[~np.isnan(features_all).any(axis=1)]
    features_all = features_all[~np.isnan(features_all).any(axis=1), :]
    features_all = np.nan_to_num(features_all, neginf=0, posinf=0)
    if os.path.exists(gedi_dir + 'near_atl08.csv'):
        df = pd.read_csv(gedi_dir + 'near_atl08.csv')
        data = df.to_numpy()
        gedi_valid_idx = data[:, 1].astype(np.int)
        gedi_z_atl08_interp = data[:, 2]
    else:
        dist, idx = general.nn(gedi_XY, atl08_XY, 6)
        gedi_valid_idx = np.where(dist[:, 0] <= distance_threshold)[0]
        gedi_z_atl08_interp = general.simple_idw(dist[gedi_valid_idx, :], atl08_orthoH[idx[gedi_valid_idx, :]])
        # gedi_z_atl08_interp = general.natural_neighbor_points(atl08_XY, atl08_orthoH, gedi_XY[gedi_valid_idx, :])
        valid_label_idx = np.where(~np.isnan(gedi_z_atl08_interp))[0]
        gedi_valid_idx = gedi_valid_idx[valid_label_idx]
        gedi_z_atl08_interp = gedi_z_atl08_interp[valid_label_idx]
        tmp = np.stack((gedi_valid_idx, gedi_z_atl08_interp), axis=-1)
        df = pd.DataFrame(tmp, columns=['gedi index', 'interpolated orthometric height'])
        df.to_csv(gedi_dir + 'near_atl08.csv')
    labels = gedi_z_atl08_interp
    features = features_all[gedi_valid_idx, :]
    # gedi_tmp0 = np.stack((features[:, 0], features[:, 1], labels), axis=-1)
    # atl08_tmp0 = atl08_tmp[:, 2:5]
    # gedi_atl08_tmp0 = np.append(gedi_tmp0, atl08_tmp0, axis=0)
    # df0 = pd.DataFrame(gedi_atl08_tmp0, columns=['X', 'Y', 'ellipsoid height'])
    # df0.to_csv(gedi_dir+'gedi_near_atl08&atl08.csv')
    print('Valid GEDI number near ATL08: ', len(gedi_valid_idx), gedi_num_pts)

    if "random_forest" in regress_method:
        # regression.RF_tunning(features, labels, np.arange(700, 1000, 50))
        regressor, scaler, prediction, errors, \
        train_features, test_features, train_labels, test_labels, train_indices, test_indices \
            = regress(gedi_dir, features, labels, feature_list, distance_threshold, regress_method)
        prediction_for_all = regressor.predict(scaler.transform(features_all))
    else:
        regressor, prediction, errors, \
        train_features, test_features, train_labels, test_labels, train_indices, test_indices \
            = regress(gedi_dir, features, labels, feature_list, distance_threshold, regress_method)
        prediction_for_all = regressor.predict(features_all)
    # tree_visual(regressor, 6, train_features, train_labels, feature_list, "None")

    # errors_original = features[train_indices, 2] - gedi_orthoH_3dep[gedi_valid_idx[train_indices]].flatten()
    # errors_prediction = regressor.predict(train_features) - gedi_orthoH_3dep[gedi_valid_idx[train_indices]].flatten()
    # errors_reference_data = atl08_orthoH - atl08_orthoH_3dep
    # regression.plot_regression_results(errors_original, errors_prediction, errors_reference_data,
    #                                    gedi_dir + 'regression model\\' + regress_method + '\\train_dataset_' +
    #                                    str(distance_threshold) + '.png')

    # errors_original = features[test_indices, 5] - gedi_orthoH_3dep[gedi_valid_idx[test_indices]].flatten()
    # errors_prediction = prediction - gedi_orthoH_3dep[gedi_valid_idx[test_indices]].flatten()
    # regression.plot_regression_results(errors_original, errors_prediction, errors_reference_data,
    #                                    gedi_dir + 'regression model\\' + regress_method + '\\test_dataset_' +
    #                                    str(distance_threshold) + '.png')

    prediction_for_all = prediction_for_all.reshape((prediction_for_all.shape[0], 1))
    gedi_orthoH_3dep = gedi_orthoH_3dep.reshape((gedi_orthoH_3dep.shape[0], 1))

    regression.plot_regression_results_R(gedi_orthoH_3dep, prediction_for_all,
                                         "Comparison between prediction and 3DEP DEM by " + regress_method,
                                         scores=0, elapsed_time=0)

    gedi_tmp = np.concatenate((gedi_tmp, gedi_orthoH_srtm90.reshape((len(gedi_orthoH_srtm90), 1)),
                               prediction_for_all.reshape((len(prediction_for_all), 1))), axis=1)
    df = pd.DataFrame(gedi_tmp,
                      columns=["lon", "lat", "X", "Y", "OH_original", "OH_3dep", 'OH_srtm90', 'OH_prediction'])

    df.to_csv(gedi_dir + 'regression_model\\' + regress_method + '\\prediction_all_' +
              str(distance_threshold) + '.csv')
    atl08_tmp = np.concatenate((atl08_tmp, atl08_orthoH_srtm90.reshape((len(atl08_orthoH_srtm90), 1)),
                                atl08_tmp[:, 4].reshape((len(atl08_tmp), 1))), axis=1)
    gedi_atl08_tmp = np.append(gedi_tmp, atl08_tmp, axis=0)
    df = pd.DataFrame(gedi_atl08_tmp,
                      columns=["lon", "lat", "X", "Y", "OH_original", "OH_3dep", 'OH_srtm90', 'OH_prediction'])
    df["error"] = df["OH_prediction"] - df["OH_srtm90"]
    df.to_csv(gedi_dir + 'regression_model\\' + regress_method + '\\prediction_all_gedi_and_atl08_' +
              str(distance_threshold) + '.csv')
    df_filter_idx = general.filter(df["OH_prediction"].to_numpy(), df["OH_srtm90"].to_numpy(), [0.2, 0.8])
    df_filtered = df.iloc[np.where(df_filter_idx == True)[0], :]
    df_filtered.to_csv(gedi_dir + 'regression_model\\' + regress_method + '\\prediction_all_gedi_and_atl08_filtered_' +
                       str(distance_threshold) + '.csv')
    # general.data_analysis("GEDI predicted", gedi_tmp[:, [0, 1, 2, 3, 7, 5]],
    #                       ["lon", "lat", "X", "Y", "predicted height", "reference height"],
    #                       plot=True,
    #                       save_path=gedi_dir + 'regression model\\' + regress_method + '\\prediction_all_' +
    #                                 str(distance_threshold) + '.png')

    cell_size = 90
    XY = np.stack((df_filtered["X"].to_numpy(), df_filtered["Y"].to_numpy()), axis=-1)
    gediX, gediY, gediZ = general.rasterize(cell_size, XY, df_filtered["OH_prediction"].to_numpy(),
                                            gedi_dir + 'regression_model\\' + regress_method + '\\' + "raster " + str(
                                                distance_threshold) + '_DEM' + str(
                                                cell_size)
                                            + ".png", out_epsg)
    cell_size = 30
    gediX, gediY, gediZ = general.rasterize(cell_size, XY, df_filtered["OH_prediction"].to_numpy(),
                                            gedi_dir + 'regression_model\\' + regress_method + '\\' + "raster " + str(
                                                distance_threshold) + '_DEM' + str(
                                                cell_size)
                                            + ".png", out_epsg)


def reproj_match(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster.

    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,  # input CRS
                dst_crs,  # output CRS
                match.width,  # input width
                match.height,  # input height
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})
        print("Coregistered to shape:", dst_height, dst_width, '\n Affine', dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def dem_match(target_fpath, reference_fpath, outfile):
    CR = COREG(reference_fpath, target_fpath, path_out=outfile)
    CR.calculate_spatial_shifts()
    CR.correct_shifts()
    return None


def DEM_diff(dem_dir, reference_dem_dir, distance_threshold, cell_size, outfile):
    if "srtm" in dem_dir:
        dem_path = dem_dir
        if os.path.exists(reference_dem_dir.replace('.tif', '_reg2_srtm.tif')):
            with rasterio.open(reference_dem_dir.replace('.tif', '_reg2_srtm.tif')) as src:
                reference_dem = src.read(1)
        else:
            reproj_match(infile=reference_dem_dir,
                         match=dem_path,
                         outfile=reference_dem_dir.replace('.tif', '_reg2_srtm.tif'))
            with rasterio.open(reference_dem_dir.replace('.tif', '_reg2_srtm.tif')) as src:
                reference_dem = src.read(1)
    else:
        dem_path = dem_dir + "raster_" + str(distance_threshold) + '_DEM' + str(cell_size) + '.tif'
        if os.path.exists(reference_dem_dir.replace('.tif', '_reg2_' + str(cell_size) + 'm.tif')):
            with rasterio.open(reference_dem_dir.replace('.tif', '_reg2_' + str(cell_size) + 'm.tif')) as src:
                reference_dem = src.read(1)
        else:
            # reproj_match(infile=reference_dem_dir,
            #              match=dem_path,
            #              outfile=reference_dem_dir.replace('.tif', '_reg2_' + str(cell_size) + 'm.tif'))
            # with rasterio.open(reference_dem_dir) as src:
            #     print("original 3DEP DEM crs: ", src.crs)
                # reference_dem = src.read(1)
                # print("original 3DEP DEM shape: ", reference_dem.shape)
            dem_match(reference_dem_dir, dem_path,
                      outfile=reference_dem_dir.replace('.tif', '_reg2_' + str(cell_size) + 'm.tif'))

            with rasterio.open(reference_dem_dir.replace('.tif', '_reg2_' + str(cell_size) + 'm.tif')) as src:
                print("transformed 3DEP DEM crs: ", src.crs)
                reference_dem = src.read(1)
                print("transformed 3DEP DEM shape: ", reference_dem.shape)

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        dst_crs = src.crs
        dst_transform, dst_width, dst_height = calculate_default_transform(
            dst_crs,  # input CRS
            dst_crs,  # output CRS
            src.width,  # input width
            src.height,  # input height
            *src.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
        )
    if 'tipp' in dem_path:
        dem_off = dem[::-1, :] - 0.3048 * reference_dem
    else:
        dem_off = dem[::-1, :] - reference_dem
    dem_off = dem_off.reshape(dem.shape)
    with rasterio.open(dem_dir + outfile, 'w',
                       height=dst_height, width=dst_width, count=1, dtype=dem_off.dtype, transform=dst_transform,
                       crs=dst_crs) as dst:
        dst.write(dem_off, 1)


def evaluate(indir, out_fname):
    res_srtm_path = indir + "diff_SRTM90_3DEP.tif"
    res_NED_path = indir + "diff_NED30_3DEP.tif"
    res_rf90_path = indir + "diff_RF90_3DEP.tif"
    res_rf30_path = indir + "diff_RF30_3DEP.tif"
    # print("Evaluating SRTM 90:")
    # stats_srtm = general.raster_statistics(res_srtm_path)
    # print("Evaluating NED 30:")
    # stats_NED = general.raster_statistics(res_NED_path)
    print("Evaluating RF 90:")
    stats_rf90 = general.raster_statistics(res_rf90_path)
    print("Evaluating RF 30:")
    stats_rf30 = general.raster_statistics(res_rf30_path)
    stats = np.stack((stats_rf30, stats_rf90), axis=0)
    df = pd.DataFrame(stats, columns=["mean", "median", "std", "skew", "kurtosis"])
    df = df.rename(index={0: 'rf30', 1: 'rf90'})
    df.to_excel(indir + out_fname)



